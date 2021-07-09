import numpy as np
import torch

def get_sinusoid_encoding_table(length, channels, min_timescale=1, max_timescale=1e4):
    """ Compute the sinusoidal position encoding

    :param length: int, the length (time steps) of the encoding
    :param channels: int, the channels of the encoding
    :param min_timescale: int, min time step
    :param min_timescale: int, max time step


    :returns: A FloatTensor with shape [length, channels], denoting the position encoding
    """

    position = np.arange(length)
    num_timescales = channels // 2

    log_timescale_increment = (
            np.log(float(max_timescale) / float(min_timescale)) /
            (num_timescales - 1)
    )
    inv_timescales = min_timescale * np.exp(
        np.arange(num_timescales) * -log_timescale_increment
    )
    scaled_time = np.expand_dims(position, 1) * np.expand_dims(inv_timescales, 0)
    signal = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
    signal = np.pad(signal, [[0, 0], [0, channels % 2]])
    return torch.FloatTensor(signal)


def attention_bias(inputs, mode, inf=-1e20):
    """ A bias tensor used in attention mechanism
    :param inputs: A tensor
    :param mode: either "causal" or "masking"
    :param inf: A float, denoting an inf value

    :returns: A tensor with shape [batch, heads, queries, memories]
    """

    if mode == "causal":
        result = torch.triu(torch.ones([inputs, inputs]), diagonal=1) * inf
        return result.reshape([1, 1, inputs, inputs])
    elif mode == "masking":
        result = (1.0 - inputs.float()) * inf
        return result.unsqueeze(1).unsqueeze(1)
    else:
        raise ValueError("Unknown mode %s" % mode)


def impute(x, lengths, channels_last=True):
    """ Set elements of a batch of a sequence of tensors to zero according to sequence lengths.
    :param x: A tensor with shape [batch, time_step, ...] or [batch, ..., time_step]
    :param lengths: A tensor with shape [batch]
    :param channels_last: A bool. If true, the time_step dimension is the second dimension, otherwise the last.

    :returns: A tensor with the same shape of x, with elements time_step > corresponding length set to 0.
    """

    if channels_last:
        max_length = x.shape[1]
    else:
        max_length = x.shape[-1]
    mask = torch.arange(max_length, device=lengths.device)[None, :] < lengths[:, None]  # [B, T]
    for _ in range(len(x.shape) - 2):
        if channels_last:
            mask = mask.unsqueeze(-1)
        else:
            mask = mask.unsqueeze(1)
    return x * mask


def mask_reduce(loss, lengths, per_sample=False):
    """ Reduce a batch of sequences according to the lengths of each sequence
    :param loss: A tensor with shape [batch, time_step]
    :param lengths: A tensor with shape [batch]
    :param per_sample: A bool.

    :returns: If per_sample, return a tensor with shape [batch], the loss averaged over the valid elements on each
    sequence; otherwise, return a scalar, the loss averaged over the entire batch.
    """

    if per_sample:
        loss = impute(loss, lengths).sum(-1) / lengths  # [B]
    else:
        loss = impute(loss, lengths).sum() / lengths.sum()
    return loss


def truncated_normal(tensor, mean=0, std=0.5):
    """ Truncated normal distribution, similar to tf.random.truncated_normal
    :param tensor: A tensor of arbitrary shape.
    :param mean: mean of the distribution
    :param std: std of the distribution

    :returns: A tensor with the same shape of x, with values draw from the designated truncated normal.
    """

    with torch.no_grad():
        size = tensor.shape
        tmp = tensor.new_empty(size + (8,)).normal_(mean=mean, std=std)
        valid = (tmp < 2 * std) & (tmp > -2 * std)
        ind = valid.max(-1, keepdim=True)[1]
        tmp = tmp.gather(-1, ind).squeeze(-1)
        return tmp


def variance_scaling_initializer(tensor, factor=2.0):
    """ Variance scaling initializing using FAN_AVG, similar to the one in TF
    :param tensor: A tensor of arbitrary shape.
    :param fan_in: int
    :param fan_out: int
    :param factor: A scalar, the scaling factor.

    :returns: A tensor with the same shape of x, initialized by truncated normal with
    mean=0, std=sqrt(1.3 * factor / ((fan_in + fan_out) / 2)).
    """
    fan_in = tensor.shape[1]
    fan_out = tensor.shape[0]
    for dim in tensor.shape[2:]:
        fan_in *= dim
        fan_out *= dim
    n = (fan_in + fan_out) / 2
    return truncated_normal(tensor, std=np.sqrt(1.3 * factor / n))
