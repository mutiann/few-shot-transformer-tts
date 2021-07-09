import torch
from torch import nn
from torch.nn import functional as F
from transformer.attention import MultiheadAttention
from transformer.common import *


class FFNLayer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.1):
        super(FFNLayer, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout_rate)
        self.output_layer = nn.Linear(hidden_size, output_size, bias=False)

    def forward(self, inputs):
        hidden = self.input_layer(inputs)
        hidden = F.relu(hidden)
        hidden = self.dropout(hidden)
        outputs = self.output_layer(hidden)
        return outputs


class TransformerEncoder(nn.Module):
    def __init__(self, input_size, hparams):
        super(TransformerEncoder, self).__init__()
        self.self_attentions = nn.ModuleList()
        self.attn_layer_norms = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.ffn_layer_norms = nn.ModuleList()
        self.pe_scale = nn.Parameter(torch.tensor(1.0))
        self.dropout = nn.Dropout(hparams.transformer_dropout_rate)

        hidden_size = hparams.encoder_hidden
        for layer in range(hparams.n_encoder_layer):
            in_size = input_size if layer == 0 else hidden_size
            self.attn_layer_norms.append(nn.LayerNorm(in_size, eps=1e-6))
            self.self_attentions.append(MultiheadAttention(key_size=in_size,
                                                           value_size=in_size,
                                                           is_self_attention=True,
                                                           num_heads=hparams.n_attention_head,
                                                           dropout_rate=hparams.transformer_dropout_rate))

            self.ffn_layer_norms.append(nn.LayerNorm(hidden_size, eps=1e-6))
            self.ffn_layers.append(FFNLayer(hidden_size, hidden_size * 4, hidden_size,
                                            dropout_rate=hparams.transformer_dropout_rate))

        self.output_layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)

    def prepare_inputs(self, inputs, input_lengths):
        mask = torch.arange(inputs.shape[1], device=input_lengths.device)[None, :] < input_lengths[:, None]
        inputs = inputs * mask.unsqueeze(-1)
        bias = attention_bias(mask, "masking").to(inputs.device)
        pe = get_sinusoid_encoding_table(length=inputs.shape[1], channels=inputs.shape[2]).to(inputs.device)
        inputs += pe * self.pe_scale
        inputs = self.dropout(inputs)
        return inputs, bias

    def forward(self, inputs, input_lengths):
        x, bias = self.prepare_inputs(inputs, input_lengths)
        for i in range(len(self.self_attentions)):
            y = self.self_attentions[i](queries=self.attn_layer_norms[i](x),
                                        memories=None,
                                        bias=bias)["outputs"]
            x = x + self.dropout(y)

            y = self.ffn_layers[i](self.ffn_layer_norms[i](x))
            x = x + self.dropout(y)
        outputs = self.output_layer_norm(x)
        return outputs


class TransformerDecoder(nn.Module):
    def __init__(self, input_size, hparams):
        super(TransformerDecoder, self).__init__()
        self.self_attentions = nn.ModuleList()
        self.attn_layer_norms = nn.ModuleList()
        self.encdec_attentions = nn.ModuleList()
        self.encdec_layer_norms = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.ffn_layer_norms = nn.ModuleList()
        self.pe_scale = nn.Parameter(torch.tensor(1.0))
        self.dropout = nn.Dropout(hparams.transformer_dropout_rate)

        hidden_size = hparams.decoder_hidden
        for layer in range(hparams.n_decoder_layer):
            in_size = input_size if layer == 0 else hidden_size

            self.attn_layer_norms.append(nn.LayerNorm(in_size, eps=1e-6))
            self.self_attentions.append(MultiheadAttention(key_size=in_size,
                                                           value_size=in_size,
                                                           is_self_attention=True,
                                                           num_heads=hparams.n_attention_head,
                                                           dropout_rate=hparams.transformer_dropout_rate))

            self.encdec_layer_norms.append(nn.LayerNorm(in_size, eps=1e-6))
            self.encdec_attentions.append(MultiheadAttention(key_size=hidden_size,
                                                             value_size=hidden_size,
                                                             is_self_attention=False,
                                                             num_heads=hparams.n_attention_head,
                                                             dropout_rate=hparams.transformer_dropout_rate))

            self.ffn_layer_norms.append(nn.LayerNorm(hidden_size, eps=1e-6))
            self.ffn_layers.append(FFNLayer(hidden_size, hidden_size * 4, hidden_size,
                                            dropout_rate=hparams.transformer_dropout_rate))

        self.output_layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)

    def prepare_inputs(self, inputs, targets, input_lengths, target_lengths):
        mask = torch.arange(inputs.shape[1], device=input_lengths.device)[None, :] < input_lengths[:, None]

        enc_bias = attention_bias(mask, "masking").to(inputs.device)
        dec_bias = attention_bias(targets.shape[1], "causal").to(inputs.device)

        targets = impute(targets, target_lengths)
        targets = torch.cat([torch.zeros([targets.shape[0], 1, targets.shape[2]], device=targets.device), targets],
                            dim=1)[:, :-1]
        pe = get_sinusoid_encoding_table(length=targets.shape[1], channels=targets.shape[2]).to(targets.device)
        targets += pe * self.pe_scale

        targets = self.dropout(targets)
        return inputs, targets, dec_bias, enc_bias

    def forward(self, inputs, targets, input_lengths, target_lengths):
        memory, x, query_bias, memory_bias = self.prepare_inputs(inputs, targets, input_lengths, target_lengths)
        attn_align = []
        encdec_align = []
        for i in range(len(self.self_attentions)):
            y = self.self_attentions[i](queries=self.attn_layer_norms[i](x),
                                        memories=None,
                                        bias=query_bias)
            attn_align.append(y['align'])
            x = x + self.dropout(y["outputs"])

            y = self.encdec_attentions[i](queries=self.encdec_layer_norms[i](x),
                                          memories=memory,
                                          bias=memory_bias)
            encdec_align.append(y['align'])
            x = x + self.dropout(y["outputs"])

            y = self.ffn_layers[i](self.ffn_layer_norms[i](x))
            x = x + self.dropout(y)
        outputs = self.output_layer_norm(x)

        outputs = impute(outputs, target_lengths)
        return outputs, {'self': attn_align, 'encdec': encdec_align}
