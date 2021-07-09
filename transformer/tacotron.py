import torch
from torch import nn
from torch.nn import functional as F
from transformer.modules import TransformerEncoder, TransformerDecoder
from transformer.common import impute, mask_reduce, truncated_normal, variance_scaling_initializer


class Encoder(nn.Module):
    def __init__(self, hparams):
        super(Encoder, self).__init__()
        self.hparams = hparams
        self.embed = nn.Embedding(hparams.vocab_size, hparams.embed_size)
        if hparams.multi_speaker:
            self.speaker_embed = nn.Embedding(hparams.max_num_speaker, hparams.speaker_embedding_size)
            self.speaker_layer = nn.Linear(hparams.speaker_embedding_size, hparams.speaker_embedding_size)
        if hparams.multi_lingual:
            self.language_embed = nn.Linear(hparams.max_num_language, hparams.language_embedding_size, bias=False)
            self.language_layer = nn.Linear(hparams.language_embedding_size, hparams.language_embedding_size)
        self.encoder = TransformerEncoder(hparams.embed_size, hparams)

    def get_language_embed(self, x):
        x = self.language_embed(x)
        x = self.language_layer(x)
        x = F.softsign(x)
        return x

    def get_speaker_embed(self, x):
        x = self.speaker_embed(x)
        x = self.speaker_layer(x)
        x = F.softsign(x)
        return x

    def forward(self, inputs, input_lengths, input_spk_ids=None, input_language_vecs=None):
        inputs = self.embed(inputs)
        encoder_outputs = self.encoder(inputs, input_lengths)
        if self.hparams.multi_speaker:
            spk_embed = self.get_speaker_embed(input_spk_ids)
            spk_embed = spk_embed.unsqueeze(1).repeat([1, inputs.shape[1], 1])
            encoder_outputs = torch.cat([encoder_outputs, spk_embed], dim=-1)
        if self.hparams.multi_lingual:
            lan_embed = self.get_language_embed(input_language_vecs)  # [B, H]
            lan_embed = lan_embed.unsqueeze(1).repeat([1, inputs.shape[1], 1])
            encoder_outputs = torch.cat([encoder_outputs, lan_embed], dim=-1)
        return encoder_outputs


class DecoderPrenet(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, dropout_rate):
        super(DecoderPrenet, self).__init__()
        self.dense0 = nn.Linear(in_size, hidden_size)
        self.dense1 = nn.Linear(hidden_size, hidden_size)
        self.dense_final = nn.Linear(hidden_size, out_size, bias=False)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.dense0(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.dense1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.dense_final(x)
        return x


class Postnet(nn.Module):
    def __init__(self, hparams):
        super(Postnet, self).__init__()
        self.conv_layers = nn.ModuleList()
        self.batchnorm_layers = nn.ModuleList()
        self.dropout = nn.Dropout(hparams.decoder_dropout_rate)
        hidden = hparams.postnet_hidden
        for i in range(hparams.n_postnet_layer):
            in_size = hparams.num_mels if i == 0 else hidden
            out_size = hparams.num_mels if i == hparams.n_postnet_layer - 1 else hidden
            self.conv_layers.append(nn.Conv1d(in_size, out_size, 5, stride=1, padding=2, bias=False))
            self.batchnorm_layers.append(nn.BatchNorm1d(out_size))

    def forward(self, inputs, input_lengths):
        x = inputs.transpose(1, 2)  # NWC -> NCW
        for i in range(len(self.conv_layers)):
            x = impute(x, input_lengths, channels_last=False)
            x = self.conv_layers[i](x)
            x = self.batchnorm_layers[i](x)
            if i != len(self.conv_layers) - 1:
                x = F.tanh(x)
            x = self.dropout(x)
        return x.transpose(2, 1)


class Decoder(nn.Module):
    def __init__(self, hparams):
        super(Decoder, self).__init__()
        in_size = hparams.encoder_hidden
        if hparams.multi_speaker:
            in_size += hparams.speaker_embedding_size
        if hparams.multi_lingual:
            in_size += hparams.language_embedding_size
        self.prenet = DecoderPrenet(hparams.num_mels, hparams.prenet_hidden,
                                    hparams.decoder_hidden, hparams.decoder_dropout_rate)
        self.decoder = TransformerDecoder(in_size, hparams)
        self.mel_net = nn.Linear(hparams.decoder_hidden, hparams.num_mels, bias=False)
        self.stop_net = nn.Linear(hparams.decoder_hidden, 1)

    def forward(self, encoder_outputs, input_lengths, targets, target_lengths, leave_one=False):
        dec_inputs = self.prenet(targets)
        if leave_one:
            dec_inputs[:, -1] *= 0
        outputs, align = self.decoder(encoder_outputs, dec_inputs, input_lengths, target_lengths)
        mels = self.mel_net(outputs)
        mels = impute(mels, target_lengths)
        stop_logits = self.stop_net(outputs.detach()).squeeze(-1)
        stop_logits = impute(stop_logits, target_lengths)
        return mels, stop_logits, align


class Tacotron(nn.Module):
    def __init__(self, hparams):
        super(Tacotron, self).__init__()
        self.encoder = Encoder(hparams)
        self.decoder = Decoder(hparams)
        self.postnet = Postnet(hparams)

    def forward(self, inputs, input_lengths, mel_targets, target_lengths, input_spk_ids,
                input_language_vecs, **kwargs):
        enc_outputs = self.encoder(inputs, input_lengths, input_spk_ids, input_language_vecs)
        mel_bef, stop_logits, alignments = \
            self.decoder(enc_outputs, input_lengths, mel_targets, target_lengths)
        mel_res = self.postnet(mel_bef, target_lengths)
        mel_aft = mel_bef + mel_res
        return {'mel_bef': mel_bef, 'mel_aft': mel_aft, 'stop_logits': stop_logits, 'alignments': alignments}


def compute_loss(model, mel_targets, target_lengths, outputs, hparams):
    bef_loss = F.mse_loss(outputs['mel_bef'], mel_targets, reduction='none').mean(-1)
    bef_loss = mask_reduce(bef_loss, target_lengths)

    aft_loss = F.mse_loss(outputs['mel_aft'], mel_targets, reduction='none').mean(-1)
    aft_loss_samplewise = mask_reduce(aft_loss, target_lengths, per_sample=True)
    aft_loss = mask_reduce(aft_loss, target_lengths)

    l2_reg = hparams.reg_weight * sum((p ** 2).sum() / 2 for n, p in model.named_parameters() if
                                      'weight' in n and 'layer_norm' not in n and 'batchnorm' not in n
                                      and 'encoder.speaker_embed' not in n and 'encoder.embed' not in n)

    stop_target = (torch.arange(mel_targets.shape[1], device=target_lengths.device)[None, :]
                   == target_lengths[:, None] - 1).float()
    ce_loss = F.binary_cross_entropy_with_logits(outputs['stop_logits'], stop_target, reduction='none',
                                                 pos_weight=torch.FloatTensor([5]).to(stop_target.device))
    ce_loss = mask_reduce(ce_loss, target_lengths)

    mse_loss = (bef_loss + aft_loss) / 2

    loss = bef_loss + aft_loss + l2_reg + ce_loss
    return {'loss': loss, 'bef_loss': bef_loss, 'aft_loss': aft_loss, 'aft_losses': aft_loss_samplewise,
            'mse_loss': mse_loss, 'l2': l2_reg, 'stop_loss': ce_loss}


def initialize_variables(model):
    state_dict = model.state_dict()
    updates = {}
    for name, tensor in state_dict.items():
        if name == 'encoder.embed.weight':
            updates[name] = torch.normal(mean=0, std=1, size=tensor.shape).to(tensor.device)
        elif name in ['encoder.speaker_embed.weight', 'encoder.language_embed.weight']:
            updates[name] = truncated_normal(tensor, mean=0, std=0.5).to(tensor.device)
        elif ('weight' in name) and ('layer_norm' not in name and 'batchnorm' not in name):
            updates[name] = variance_scaling_initializer(tensor).to(tensor.device)
        elif 'bias' in name:
            updates[name] = torch.zeros_like(tensor)
    model.load_state_dict(updates, strict=False)


def learning_rate_schedule(global_step, hp):
    step = max(global_step - hp.warmup_steps, 0)
    lr_rate = hp.lr_decay_rate ** (step / hp.lr_decay_step)
    return max(hp.min_lr / hp.max_lr, lr_rate)
