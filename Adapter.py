#coding: utf-8
import sys
import torch
from transformers import BertModel, BertConfig
from torch import nn
import torch.nn.functional as F

from transformers.models.roberta.modeling_roberta import RobertaModel,RobertaSelfOutput,RobertaEncoder,RobertaOutput,RobertaLayer,RobertaAttention,RobertaForSequenceClassification,RobertaForMaskedLM
from typing import List, Optional, Tuple, Union

class Adapter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = torch.nn.Linear(config.hidden_size, config.adapter_size)
        self.fc2 = torch.nn.Linear(config.adapter_size, config.hidden_size)
        self.activation = torch.nn.ReLU()

    def forward(self, x, add_residual=True):
        residual = x
        h = self.activation(self.fc1(x))
        h = self.activation(self.fc2(h))
        if add_residual:
            output = residual + h
        else:
            output = h

        return output

class MyRobertaSelfOutput(RobertaSelfOutput):
    def __init__(self, config):
        super().__init__(config)
        self.adapter = Adapter(config)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.adapter(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class MyRobertaOutput(RobertaOutput):
    def __init__(self, config):
        super().__init__(config)
        self.adapter = Adapter(config)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.adapter(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class MyRobertaAttention(RobertaAttention):
    def __init__(self, config, position_embedding_type=None):
        super().__init__(config, position_embedding_type= position_embedding_type)
        self.output = MyRobertaSelfOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ):
        return super().forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask, 
        past_key_value, output_attentions)

class MyRobertaLayer(RobertaLayer):
    def __init__(self, config):
        super().__init__(config)
        self.attention = MyRobertaAttention(config)
        self.output = MyRobertaOutput(config)
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = MyRobertaAttention(config)

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
        return super().forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask, 
        past_key_value, output_attentions)

class MyRobertaEncoder(RobertaEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.layer = nn.ModuleList([MyRobertaLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ):
        return super().forward(hidden_states,attention_mask,head_mask,encoder_hidden_states,encoder_attention_mask,past_key_values,
        use_cache, output_attentions,output_hidden_states,return_dict)


class MyRobertaModel(RobertaModel):
    def __init__(self, config,add_pooling_layer=False):
        super().__init__(config,add_pooling_layer = add_pooling_layer)
        self.encoder = MyRobertaEncoder(config)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return super().forward(input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds,
        encoder_hidden_states, encoder_attention_mask, past_key_values, use_cache, output_attentions, output_hidden_states, return_dict)

class Roberta_Net_cls(RobertaForSequenceClassification):
    def __init__(self,config):
        super().__init__(config)

        self.roberta = MyRobertaModel(config)
        
        for param in self.roberta.parameters():
            param.requires_grad = False

        adaters = \
            [self.roberta.encoder.layer[layer_id].attention.output.adapter for layer_id in range(config.num_hidden_layers)] + \
            [self.roberta.encoder.layer[layer_id].attention.output.LayerNorm for layer_id in range(config.num_hidden_layers)] + \
            [self.roberta.encoder.layer[layer_id].output.adapter for layer_id in range(config.num_hidden_layers)] + \
            [self.roberta.encoder.layer[layer_id].output.LayerNorm for layer_id in range(config.num_hidden_layers)]

        for adapter in adaters:
            for param in adapter.parameters():
                param.requires_grad = True

        print('Roberta ADAPTER')

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):

        return super().forward(input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds,
        labels, output_attentions, output_hidden_states, return_dict)
    
class Roberta_Net_mlm(RobertaForMaskedLM):
    def __init__(self,config):
        super().__init__(config)

        self.roberta = MyRobertaModel(config)
        
        for param in self.roberta.parameters():
            param.requires_grad = False

        adaters = \
            [self.roberta.encoder.layer[layer_id].attention.output.adapter for layer_id in range(config.num_hidden_layers)] + \
            [self.roberta.encoder.layer[layer_id].attention.output.LayerNorm for layer_id in range(config.num_hidden_layers)] + \
            [self.roberta.encoder.layer[layer_id].output.adapter for layer_id in range(config.num_hidden_layers)] + \
            [self.roberta.encoder.layer[layer_id].output.LayerNorm for layer_id in range(config.num_hidden_layers)]

        for adapter in adaters:
            for param in adapter.parameters():
                param.requires_grad = True

        print('Roberta ADAPTER')

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return super().forward(input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds,encoder_hidden_states,
        encoder_attention_mask, labels, output_attentions, output_hidden_states, return_dict)