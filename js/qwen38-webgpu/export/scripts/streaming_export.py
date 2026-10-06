"""Memory-bounded save adapter around the official builder and ORT quantizer.

Only use if the baseline exporter exceeds this host's memory ceiling. This
serializes/quantizes one weight at a time with the official DEFAULT quantizer,
writing quantized bytes directly to their final external-data file. No model
operators, quantization settings or inference kernels are replaced.
"""
import gc
import importlib.util
import json
import os
import runpy
import sys
import math
from pathlib import Path

import onnx
import numpy as np
import onnx_ir as ir
from onnx import helper, numpy_helper, external_data_helper
from onnxruntime.quantization.matmul_nbits_quantizer import DefaultWeightOnlyQuantizer, DefaultWeightOnlyQuantConfig, QuantFormat

source = Path(importlib.util.find_spec('onnxruntime_genai.models.builder').origin).parent
sys.path.insert(0, str(source))
from builders.base import Model


def memory_bounded_save(self, out_dir):
    assert self.onnx_dtype == ir.DataType.INT4 and self.ep == 'webgpu'
    assert getattr(self, 'quantization_algo', 'default') == 'default'
    assert not getattr(self, 'int4_customized_weight_config', {})
    assert not getattr(self, 'exact_quant_overrides', {})
    assert not self.quant_attrs['use_qdq']
    self.model.graph.sort()
    Model.stamp_build_metadata(self.model)
    originals = {name: value.const_value for name, value in self.model.graph.initializers.items()}
    # Serialize graph metadata without materializing the full FP16 model.
    for name, value in self.model.graph.initializers.items():
        tensor = originals[name]
        value.const_value = ir.ExternalTensor('pending', 0, 0, tensor.dtype,
                                             shape=tensor.shape, name=name)
    proto = ir.to_proto(self.model)
    for name, value in self.model.graph.initializers.items():
        value.const_value = originals[name]
    del proto.graph.initializer[:]
    nodes = list(proto.graph.node)
    del proto.graph.node[:]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    shard_limit = int(os.environ.get('SHARD_BYTES', 1024 ** 3))
    shard_index = 0
    location = self.filename + f'.data_{shard_index:03d}'
    shard_paths = [out / location]
    stored = {}
    records = []

    with (out / location).open('wb') as data_file:
        def store(tensor):
            nonlocal data_file, location, shard_index
            if tensor.name in stored:
                return
            raw = tensor.raw_data or numpy_helper.to_array(tensor).tobytes()
            if len(raw) > shard_limit:
                raise ValueError(f'Tensor exceeds shard limit: {tensor.name}, {len(raw)} bytes')
            padding = (-data_file.tell()) % 64
            if data_file.tell() + padding + len(raw) > shard_limit:
                data_file.close()
                shard_index += 1
                location = self.filename + f'.data_{shard_index:03d}'
                shard_paths.append(out / location)
                data_file = (out / location).open('wb')
                padding = 0
            data_file.write(b'\0' * padding)
            offset = data_file.tell()
            data_file.write(raw)
            length = len(raw)
            if not tensor.HasField('raw_data'):
                tensor.raw_data = raw
            external_data_helper.set_external_data(tensor, location=location, offset=offset, length=length)
            tensor.ClearField('raw_data')
            for field in ['float_data', 'int32_data', 'int64_data', 'double_data', 'uint64_data', 'string_data']:
                tensor.ClearField(field)
            tensor.data_location = onnx.TensorProto.EXTERNAL
            # A child protobuf retains its parent graph's allocation arena.
            # Keeping the mini-graph initializer would therefore retain the
            # original FP16 payload even after ClearField('raw_data'). Copy only
            # the small external-data descriptor into an independent message.
            descriptor = onnx.TensorProto()
            descriptor.CopyFrom(tensor)
            stored[tensor.name] = descriptor

        excluded = set(self.quant_attrs['nodes_to_exclude'])
        ops = set(self.quant_attrs['op_types_to_quantize'])
        for index, node in enumerate(nodes):
            if node.op_type not in ops or node.name in excluded:
                proto.graph.node.append(node)
                continue
            weight_name = node.input[0 if node.op_type == 'Gather' else 1]
            if weight_name not in originals:
                proto.graph.node.append(node)
                continue
            tensor = originals[weight_name]
            temporary_weight = None
            elements = math.prod(int(dim) for dim in tensor.shape)
            if elements * tensor.dtype.bitwidth // 8 >= 2**31:
                # Protobuf cannot copy a >2 GiB message into a GraphProto.
                # The official quantizer can consume external-data descriptors.
                temporary_weight = out / '.quantize-weight.tmp'
                array = np.ascontiguousarray(tensor.numpy())
                array.tofile(temporary_weight)
                del array
                weight = ir.to_proto(ir.ExternalTensor(
                    str(temporary_weight), 0, temporary_weight.stat().st_size,
                    tensor.dtype, shape=tensor.shape, name=weight_name))
            else:
                weight = ir.to_proto(tensor)
            mini = helper.make_graph([node], 'one_weight', [], [], [weight])
            del weight
            quantizer = DefaultWeightOnlyQuantizer(DefaultWeightOnlyQuantConfig(
                bits=self.quant_attrs['bits'], block_size=self.quant_attrs['matmul_block_size'],
                is_symmetric=self.quant_attrs['is_symmetric'],
                accuracy_level=self.quant_attrs['accuracy_level'],
                quant_format=QuantFormat.QOperator, op_types_to_quantize=tuple(ops)))
            result_nodes = quantizer.quantize(node, [mini])
            if temporary_weight:
                temporary_weight.unlink()
            assert all(n.op_type != node.op_type for n in result_nodes), node.name
            proto.graph.node.extend(result_nodes)
            result = mini
            used = {name for n in result_nodes for name in n.input}
            for value in result.initializer:
                if value.name in used:
                    store(value)
            records.append({'name': node.name, 'weight': weight_name, 'shardBytes': data_file.tell()})
            print(f'Stream-quantized {node.name} ({index + 1}/{len(nodes)}), {data_file.tell():,} bytes', flush=True)
            # LazyTensor caches its cast result; release the temporary FP16 copy.
            if isinstance(tensor, ir.LazyTensor):
                tensor._tensor = None
            del mini, quantizer, result, result_nodes, tensor
            gc.collect()

        # Emit only initializers still referenced by the transformed graph.
        used = {name for node in proto.graph.node for name in node.input}
        for name in sorted(used):
            if name in originals and name not in stored:
                tensor = originals[name]
                store(ir.to_proto(tensor))
                if isinstance(tensor, ir.LazyTensor):
                    tensor._tensor = None
                del tensor
                gc.collect()
        proto.graph.initializer.extend(stored.values())
    data_file.close()
    onnx.save_model(proto, out / self.filename)
    notes = out.parent / (out.name + ".reports")
    notes.mkdir(parents=True, exist_ok=True)
    (notes / "streaming-quantization.json").write_text(json.dumps({
        'adapter': 'per-weight serialization with official DEFAULT MatMulNBitsQuantizer',
        'settings': self.quant_attrs, 'quantizedNodes': records,
        'externalBytes': sum(path.stat().st_size for path in shard_paths),
        'shards': [{'file': path.name, 'bytes': path.stat().st_size} for path in shard_paths],
    }, indent=2, default=str) + '\n')


Model.save_model = memory_bounded_save
if __name__ == '__main__':
    runpy.run_path(str(source / 'builder.py'), run_name='__main__')
