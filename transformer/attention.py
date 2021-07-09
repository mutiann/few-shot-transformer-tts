import torch
from torch import nn
from torch.nn import functional as F


def split_heads(x, num_heads):
    """ Split heads

    :param x: A tensor with shape [batch, length, channels]
    :param num_heads: An integer

    :returns: A tensor with shape [batch, heads, length, channels / heads]
    """
    assert x.shape[-1] % num_heads == 0, str(x.shape)
    return x.reshape(x.shape[:-1] + (num_heads, x.shape[-1] // num_heads)).permute(0, 2, 1, 3)


def combine_heads(x):
    """ Combine heads

    :param x: A tensor with shape [batch, heads, length, channels]

    :returns: A tensor with shape [batch, length, heads * channels]
    """
    x = x.permute([0, 2, 1, 3])
    return x.reshape(x.shape[:-2] + (x.shape[-1] * x.shape[-2],))


class MultiheadAttention(nn.Module):
    def __init__(self, key_size, value_size, is_self_attention, num_heads, dropout_rate=0.1):
        """multi-head attention.

        :param key_size: int, for the hidden size of keys
        :param value_size: int, for the hidden size of values
        :param is_self_attention: bool
        :param num_heads: int, number of heads
        :param dropout_rate: float, dropout rate for attention weights
        """
        super(MultiheadAttention, self).__init__()
        assert key_size % num_heads == 0, "key_size=%d, num_heads=%d" % (key_size, num_heads)
        assert value_size % num_heads == 0, "value_size=%d, num_heads=%d" % (value_size, num_heads)
        if is_self_attention:
            self.qkv_transform = nn.Linear(key_size, key_size * 2 + value_size, bias=False)
        else:
            self.q_transform = nn.Linear(key_size, key_size, bias=False)
            self.kv_transform = nn.Linear(key_size, key_size + value_size, bias=False)
        self.output_transform = nn.Linear(key_size, key_size, bias=False)
        self.attn_dropout = nn.Dropout(dropout_rate)
        self.num_heads = num_heads
        self.key_size = key_size
        self.value_size = value_size

    def compute_qkv(self, queries, memories):
        """Computes query, key and value.

        :param queries: A tensor with shape [batch, length_q, depth_q]
        :param memories: A tensor with shape [batch, length_m, depth_m]

        :returns: (q, k, v): [batch, length, depth] tensors
        """

        if memories is None:
            combined = self.qkv_transform(queries)
            q, k, v = torch.split(combined, [self.key_size, self.key_size, self.value_size], dim=-1)
        else:
            q = self.q_transform(queries)
            combined = self.kv_transform(memories)
            k, v = torch.split(combined, [self.key_size, self.value_size], dim=-1)

        return q, k, v

    def dot_product_attention(self, q, k, v, bias):
        """dot-product attention.

        :param q: A tensor with shape [batch, heads, length_q, depth_k]
        :param k: A tensor with shape [batch, heads, length_kv, depth_k]
        :param v: A tensor with shape [batch, heads, length_kv, depth_v]
        :param bias: A tensor for ignoring unreasonable position

        :returns: context:A tensor with shape [batch, heads, length_q, depth_v]
        :returns: align:A tensor with shape [N, head, T_enc, T_dec]
        """
        logits = torch.matmul(q, k.transpose(2, 3)) # [batch, num_heads, query_length, memory_length]
        if bias is not None:
            logits += bias
        # [batch, head, T_dec, T_enc]
        weights = F.softmax(logits, dim=-1)
        align = weights.permute(0, 1, 3, 2)  # [batch, num_heads, memory_length, query_length]
        weights = self.attn_dropout(weights)

        context = torch.matmul(weights, v)
        return context, align

    def forward(self, queries, memories, bias):
        """ Multi-head scaled-dot-product attention with input/output
            transformations.

        :param queries: A tensor with shape [batch, length_q, depth_q]
        :param memories: A tensor with shape [batch, length_m, depth_m]
        :param bias: A tensor (see attention_bias)

        :returns: A dict with the following keys:
            outputs: A tensor with shape [batch, length_q, depth_v]
            align: A tensor with shape [batch, heads, length_q, length_kv]
        """

        q, k, v = self.compute_qkv(queries, memories)

        q = split_heads(q, self.num_heads)
        k = split_heads(k, self.num_heads)
        v = split_heads(v, self.num_heads)

        key_depth_per_head = self.key_size // self.num_heads
        q = q * key_depth_per_head ** -0.5

        results, weight = self.dot_product_attention(q, k, v, bias)

        x = combine_heads(results)
        x = self.output_transform(x)

        outputs = {"outputs": x, 'align': weight}
        return outputs
