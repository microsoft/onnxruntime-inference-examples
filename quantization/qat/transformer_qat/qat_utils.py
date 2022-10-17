import torch
import math
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from typing import List, Optional, Tuple, Union

from enum import Enum

# Overall control of target order used in the quantization
class QatActivationLayout(Enum):
    ROW = 1
    COL32 = 2

qat_activation_layout = QatActivationLayout.ROW

# A simple hook class that returns the input and output of a layer during forward pass
class InputOutputRecorder():
    def __init__(self, module, module_name, record_input = False):
        self.record_input = record_input
        self.module_name = module_name
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        if self.record_input:
            self.input = input
        self.output = output

    def close(self):
        self.hook.remove()


def remove_qconfig_for_module_tree(mod, prefix):
    if isinstance(mod, torch.nn.Module) and hasattr(mod, "qconfig"):
        print(f"--------Removing qconfig for {prefix}")
        del mod.qconfig
    for name, child in mod.named_children():
        child_prefix = f"{prefix}.{name}" if prefix else name
        remove_qconfig_for_module_tree(child, child_prefix)


def remove_qconfig_subtree_in_module_type(mod, prefix, remove_type, keep_qconfig_for_root = True):
    if isinstance(mod, remove_type):
        if keep_qconfig_for_root:
            for name, child in mod.named_children():
                child_prefix = f"{prefix}.{name}" if prefix else name
                remove_qconfig_for_module_tree(child, child_prefix)
        else:
            remove_qconfig_for_module_tree(mod, prefix)
        return

    for name, child in mod.named_children():
        child_prefix = f"{prefix}.{name}" if prefix else name
        remove_qconfig_subtree_in_module_type(child, child_prefix, remove_type)


def remove_qconfig_for_module(mod, prefix, prefixes_to_remove, remove_subtree = False):
    if isinstance(mod, torch.nn.Module) and (prefix in prefixes_to_remove):
        if remove_subtree:
            remove_qconfig_for_module_tree(mod, prefix)
            return

        if hasattr(mod, "qconfig"):
            print(f"--------Removing qconfig for {prefix}")
            del mod.qconfig

    for name, child in mod.named_children():
        child_prefix = f"{prefix}.{name}" if prefix else name
        remove_qconfig_for_module(child, child_prefix, prefixes_to_remove, remove_subtree=remove_subtree)


def iterate_modules(mod, prefix, fn):
    if isinstance(mod, torch.nn.Module):
        fn(mod, prefix)
    for name, child in mod.named_children():
        child_prefix = f"{prefix}.{name}" if prefix else name
        iterate_modules(child, child_prefix, fn)

def copy_attributes(target, source, needed_attrs):
    for name in needed_attrs:
        if hasattr(source, name):
            setattr(target, name, getattr(source, name))
    
class QAwareLinear(nn.Linear):
    _FLOAT_MODULE = nn.Linear

    def __init__(self, fp_module: _FLOAT_MODULE, activation_layout : QatActivationLayout):
        super().__init__(fp_module.in_features, fp_module.out_features, fp_module.bias is not None)
        self.weight = fp_module.weight
        self.bias = fp_module.bias

        self.qconfig = fp_module.qconfig
        self.activation_layout = activation_layout
        self.weight_fakeq = self.qconfig.weight()
        if hasattr(self.weight_fakeq.activation_post_process, 'ch_axis'): # per channel
            self.weight_fakeq.ch_axis = self.weight_fakeq.activation_post_process.ch_axis = 0
        self.weight_fakeq.ch_axis = 0 # enforce the channel axis if needed in per-channel
        
        self.need_observe_input = False
        self.need_observe_output = (self.activation_layout == QatActivationLayout.COL32)
        self.input_fakeq = self.qconfig.activation()
        self.output_fakeq = self.qconfig.activation()

    def observe_input(self, obeserver : bool = True):
        self.need_observe_input = observe

    def observe_output(self, observe : bool = True):
        self.need_observe_output = observe

    def forward(self, input):
        if self.need_observe_input:
            input = self.input_fakeq(input)
        if self.activation_layout == QatActivationLayout.COL32:
            assert self.need_observe_output, 'output should be observed for COL32 layout activation'
            output = self.output_fakeq(F.linear(input, self.weight_fakeq(self.weight))) + self.bias
        else:
            output = F.linear(input, self.weight_fakeq(self.weight), self.bias)
            if self.need_observe_output:
                output = self.output_fakeq(output)
        return output

    @classmethod
    def from_float(cls, mod):
        print(f"==============Converting a {type(mod).__name__} module ==> {cls.__name__} module...")
        assert type(mod) == cls._FLOAT_MODULE, f"{cls.__name__} from_float() only works on {cls._FLOAT_MODULE.__name__}"
        assert hasattr(mod, 'qconfig'), 'Input float module must have qconfig defined'
        assert mod.qconfig, 'Input float module must have a valid qconfig'

        qaware_mod = cls(mod, qat_activation_layout)
        return qaware_mod
