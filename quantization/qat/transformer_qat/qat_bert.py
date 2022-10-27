import torch
import math
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from typing import List, Optional, Tuple, Union

from qat_utils import (
    qat_activation_layout, QatActivationLayout, QAwareLinear,
    iterate_modules, copy_attributes
)

from transformers.models.bert.modeling_bert import (
    BertOutput,
    BertSelfOutput,
    BertSelfAttention,
    BertIntermediate,
)

class BertQatOptions:
    def __init__(self,
                 quantize_attention_scores = False, # handle matmul q k
                 quantize_attention_probs = False,   # after softmax
                 quantize_context_layer = True):    # result is quantized, handle the matmul with v
        self.quantize_attention_scores = quantize_attention_scores
        self.quantize_attention_probs = quantize_attention_probs
        self.quantize_context_layer = quantize_context_layer


# could export out for user to control detail
bert_qat_option = BertQatOptions(quantize_attention_scores = True, quantize_attention_probs = True)


class QAwareBertSelfAttention(nn.Module):
    def __init__(self, fp_module, activation_layout : QatActivationLayout, qat_option : BertQatOptions):
        super().__init__()

        needed_attrs = ['num_attention_heads', 'attention_head_size', 'all_head_size',
                        'query', 'key', 'value', 'dropout', 'position_embedding_type',
                        'max_position_embeddings', 'distance_embedding', 'is_decoder']
        copy_attributes(self, fp_module, needed_attrs)

        self.qconfig = fp_module.qconfig
        self.activation_layout = activation_layout
        self.qat_option = qat_option
        self.att_score_fakeq = self.qconfig.activation() if self.qat_option.quantize_attention_scores else None
        self.att_probs_fakeq = self.qconfig.activation() if self.qat_option.quantize_attention_probs else None
        self.ctx_fakeq = self.qconfig.activation() if self.qat_option.quantize_context_layer else None

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:

        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        if self.qat_option.quantize_attention_scores:
            # The mask set could be merged with softmax, observe it here
            attention_scores = self.att_score_fakeq(attention_scores)

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        if self.qat_option.quantize_attention_probs:
            attention_probs = self.att_probs_fakeq(attention_probs)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)
        if self.qat_option.quantize_context_layer:
            context_layer = self.ctx_fakeq(context_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs

    @classmethod
    def from_float(cls, mod):
        print(f"==============Converting a {type(mod).__name__} module ==> {cls.__name__} module...")
        assert hasattr(mod, 'qconfig'), 'Input float module must have qconfig defined'
        assert mod.qconfig, 'Input float module must have a valid qconfig'
        qaware_mod = cls(mod, qat_activation_layout, bert_qat_option)
        return qaware_mod


class QAwareBertIntermediate(nn.Module):
    def __init__(self, fp_module, activation_layout: QatActivationLayout):
        super().__init__()
        copy_attributes(self, fp_module, ['dense', 'intermediate_act_fn'])
        self.qconfig = fp_module.qconfig
        self.activation_layout = activation_layout
        self.act_fakeq = self.qconfig.activation()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.act_fakeq(hidden_states)
        return hidden_states

    @classmethod
    def from_float(cls, mod):
        print(f"==============Converting a {type(mod).__name__} module ==> {cls.__name__} module...")
        assert hasattr(mod, 'qconfig'), 'Input float module must have qconfig defined'
        assert mod.qconfig, 'Input float module must have a valid qconfig'
        qaware_mod = cls(mod, qat_activation_layout)
        return qaware_mod


class QAwareBertOutput(nn.Module):
    def __init__(self, float_module, activation_layout):
        super().__init__()
        self.dense = float_module.dense
        self.LayerNorm = float_module.LayerNorm
        self.dropout = float_module.dropout

        self.qconfig = float_module.qconfig
        self.activation_layout = activation_layout
        if self.activation_layout == QatActivationLayout.ROW:
            if hasattr(self.dense, 'qconfig'):
                del self.dense.qconfig  # so that later prepare_ will not observe after dense
            self.linear_fakeq = self.qconfig.activation()

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor
        if self.activation_layout == QatActivationLayout.ROW:
            hidden_states = self.linear_fakeq(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        # LayerNorm will be observed after convert() unless it is not needed
        # so, do not add obeserver here
        return hidden_states

    @classmethod
    def from_float(cls, mod):
        print(f"==============Converting a {type(mod).__name__} module ==> {cls.__name__} module...")
        assert hasattr(mod, 'qconfig'), 'Input float module must have qconfig defined'
        assert mod.qconfig, 'Input float module must have a valid qconfig'
        qaware_mod = cls(mod, qat_activation_layout)
        return qaware_mod


class QAwareBertSelfOutput(nn.Module):
    def __init__(self, float_module, activation_layout):
        super().__init__()
        self.dense = float_module.dense
        self.LayerNorm = float_module.LayerNorm
        self.dropout = float_module.dropout
        self.qconfig = float_module.qconfig
        self.activation_layout = activation_layout
        if self.activation_layout == QatActivationLayout.ROW:
            if hasattr(self.dense, 'qconfig'):
                del self.dense.qconfig  # so that later prepare_ will not observe after dense
            self.linear_fakeq = self.qconfig.activation()

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor
        if self.activation_layout == QatActivationLayout.ROW:
            hidden_states = self.linear_fakeq(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        # LayerNorm will be observed after convert() unless it is not needed
        # so, do not add obeserver here
        return hidden_states

    @classmethod
    def from_float(cls, mod):
        print(f"==============Converting a {type(mod).__name__} module ==> {cls.__name__} module...")
        assert hasattr(mod, 'qconfig'), 'Input float module must have qconfig defined'
        assert mod.qconfig, 'Input float module must have a valid qconfig'
        qaware_mod = cls(mod, qat_activation_layout)
        return qaware_mod


# before_convert_fn(model) is called right before torch.quantization.convert(model..)
def quantize_bert_model(model, before_convert_fn = None):
    print("", f"====Quantizing bert model.........")

    qat_activation_layout = QatActivationLayout.ROW
    model.qconfig = torch.quantization.QConfig(
        activation=torch.quantization.default_weight_fake_quant,
        weight=torch.quantization.default_per_channel_weight_fake_quant) # or default_weight_fake_quant

    qmapping = {
        torch.nn.modules.linear.Linear : QAwareLinear,
        BertOutput : QAwareBertOutput,
        BertSelfOutput : QAwareBertSelfOutput,
        BertSelfAttention: QAwareBertSelfAttention,
        BertIntermediate: QAwareBertIntermediate,
    }

    # model = torch.quantization.prepare_qat(...) make things out of control, just do not use it
    torch.quantization.propagate_qconfig_(model, qconfig_dict=None)

    # add fakeq for matmul module in bert MiddleLayer projection
    def start_matmul_input_fakeq(mod, prefix):
        if prefix == 'MiddleLayer_2hiddensize_hiddensize.projection':
            mod.observe_input()
    iterate_modules(model, '', start_matmul_input_fakeq)

    if before_convert_fn is not None:
        before_convert_fn(model)

    torch.quantization.convert(model, mapping = qmapping, inplace=True, remove_qconfig=False)

    # do not use torch.quantization.prepare_() as it calls propagate_qconfig_(model, qconfig_dict=None) again
    non_leaf_module_to_observe = [QAwareLinear] if qat_activation_layout == QatActivationLayout.ROW else []
    torch.quantization.add_observer_(
        model,
        qconfig_propagation_list=[torch.nn.LayerNorm, torch.nn.GELU], # Leaf module infact :)
        non_leaf_module_list=set(non_leaf_module_to_observe))

    # model.apply(torch.quantization.enable_observer)
    print(f"====Finished quantizating bert model.........", "")
    return model
