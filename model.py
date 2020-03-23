#!/usr/bin/env python3
"""
    Adapted from: https://github.com/graykode/gpt-2-Pytorch
        (Commit: 46ae886391a94c6683be438269252c4afd5ba762)
    Original Paper and repository here: https://github.com/openai/gpt-2
"""
import copy
import math

import torch
import torch.nn as nn


def gelu(x):
    return (
        0.5
        * x
        * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    )


class PathLSTM(nn.Module):
    def __init__(self, vocab_size, n_embd):
        super(PathLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, n_embd)
        self.LSTM = nn.LSTM(n_embd, n_embd, batch_first=True)

    def forward(self, paths):
        embed = self.embedding(paths)  # bs, max_len, max_path_len, n_embd
        batch_size, bag_size, path_len, n_embd = embed.shape
        _, (h_n, _) = self.LSTM(embed.view(batch_size * bag_size, path_len, n_embd))
        return h_n.permute((1, 0, 2)).view((batch_size, bag_size, -1))


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, std_eps=1e-6):
        """Construct a layernorm module in the TF style.
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.std_eps = std_eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).std(-1, keepdim=True)
        x = (x - u) / (s + self.std_eps)
        return self.weight * x + self.bias


class Attention(nn.Module):
    def __init__(
        self, nx, n_ctx, n_head, scale=False, rel_vocab_size=None
    ):
        super(Attention, self).__init__()
        n_state = nx
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        assert n_state % n_head == 0
        self.register_buffer(
            "bias", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx)
        )
        self.n_head = n_head
        self.split_size = n_state
        self.scale = scale
        self.c_attn = nn.Linear(nx, n_state * 3)
        self.c_proj = nn.Linear(nx, n_state)

        # if rel exists
        if rel_vocab_size is not None:
            self.rel_weights = nn.Embedding(rel_vocab_size, n_head)

    def _attn(self, q, k, v, rel=None):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        nd, ns = w.size(-2), w.size(-1)
        b = self.bias[:, :, ns - nd : ns, :ns]
        w = w * b - 1e10 * (1 - b)

        # add in more tree structure
        if rel is not None:
            w = w * (rel * b)

        w = nn.Softmax(dim=-1)(w)
        return torch.matmul(w, v)

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def forward(self, x, rel=None):
        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        if rel is not None:
            rel = self.rel_weights(rel)
            rel = rel.permute(0, 3, 1, 2)

        # self attention component
        a = self._attn(query, key, value, rel)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        return a


class MLP(nn.Module):
    def __init__(self, n_state, n_embd):
        super(MLP, self).__init__()
        self.c_fc = nn.Linear(n_embd, n_state)
        self.c_proj = nn.Linear(n_state, n_embd)
        self.act = gelu

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return h2


class Block(nn.Module):
    def __init__(
        self,
        n_ctx,
        n_head,
        n_embd,
        layer_norm_epsilon,
        scale=False,
        rel_vocab_size=None,
    ):
        super(Block, self).__init__()
        self.ln_1 = LayerNorm(n_embd, std_eps=layer_norm_epsilon)
        self.attn = Attention(
            n_embd, n_ctx, n_head, scale, rel_vocab_size
        )
        self.ln_2 = LayerNorm(n_embd, std_eps=layer_norm_epsilon)
        self.mlp = MLP(4 * n_embd, n_embd)

    def forward(self, x, rel):
        a = self.attn(self.ln_1(x), rel)
        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m
        return x


class GPT2Model(nn.Module):
    def __init__(
        self,
        vocab_size,
        n_layer,
        n_embd,
        n_ctx,
        n_head,
        layer_norm_epsilon,
        root_paths,
        rel_vocab_size,
    ):
        super(GPT2Model, self).__init__()
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.n_vocab = vocab_size
        self.wte = nn.Embedding(vocab_size, n_embd)
        if root_paths:
            self.path_lstm = PathLSTM(vocab_size, n_embd)
        block = Block(
            n_ctx,
            n_head,
            n_embd,
            layer_norm_epsilon,
            scale=True,
            rel_vocab_size=rel_vocab_size,
        )
        self.h = nn.ModuleList([copy.deepcopy(block) for _ in range(n_layer)])
        self.ln_f = LayerNorm(n_embd, std_eps=layer_norm_epsilon)

    def forward(self, input_ids, rel=None, paths=None):
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_ids.size(-1))
        inputs_embeds = self.wte(input_ids)
        path_embeds = self.path_lstm(paths) if paths is not None else 0
        hidden_states = inputs_embeds + path_embeds

        for block in self.h:
            hidden_states = block(hidden_states, rel)
        hidden_states = self.ln_f(hidden_states)
        output_shape = input_shape + (hidden_states.size(-1),)
        return hidden_states.view(*output_shape)


class GPT2LMHead(nn.Module):
    def __init__(self, model_embeddings_weights, n_embd):
        super(GPT2LMHead, self).__init__()
        self.n_embd = n_embd
        self.set_embeddings_weights(model_embeddings_weights)

    def set_embeddings_weights(self, model_embeddings_weights):
        embed_shape = model_embeddings_weights.shape
        self.decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=False)
        self.decoder.weight = model_embeddings_weights  # Tied weights

    def forward(self, hidden_state):
        lm_logits = self.decoder(hidden_state)
        return lm_logits


class TransformerModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        loss_fn,
        n_layer,
        n_embd,
        n_ctx,
        n_head,
        layer_norm_epsilon,
        root_paths=False,
        rel_vocab_size=None,
    ):
        super(TransformerModel, self).__init__()
        self.transformer = GPT2Model(
            vocab_size,
            n_layer,
            n_embd,
            n_ctx,
            n_head,
            layer_norm_epsilon,
            root_paths,
            rel_vocab_size,
        )
        self.lm_head = GPT2LMHead(self.transformer.wte.weight, n_embd)
        self.loss_fn = loss_fn

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self, x, y, ext=None, rel=None, paths=None, return_loss=False
    ):
        hidden_states = self.transformer(x, rel, paths)
        y_pred = self.lm_head(hidden_states)
        if not return_loss:
            return y_pred

        # ext contains a list of idx of where to take the loss from
        # we linearize it first
        ids = []
        max_len = y.size(-1)
        for i, ext_i in enumerate(ext):
            ids += [i * max_len + j for j in range(ext_i, max_len)]
        loss = self.loss_fn(y_pred.view(-1, y_pred.size(-1))[ids], y.view(-1)[ids])
        return loss


# base RNN model
class LSTMModel(torch.nn.Module):
    def __init__(self, vocab_size, n_embd, loss_fn, n_ctx):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, n_embd)
        self.lstm = nn.LSTM(n_embd, n_embd, num_layers=1, dropout=0.5, batch_first=True)
        self.decoder = nn.Linear(n_embd, vocab_size)

        self.loss_fn = loss_fn
        self.half_ctx = int(n_ctx / 2)

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self, x, y, ext=None, rel=None, paths=None, return_loss=False
    ):
        embed = self.embedding(x)  # bs, max_len, n_embd
        self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(embed)  # bs, max_len, n_embd
        y_pred = self.decoder(lstm_out)  # bs, max_len, vocab_size
        if not return_loss:
            return y_pred

        # ext contains a list of idx of where to take the loss from
        # we linearize it first
        ids = []
        max_len = y.size(-1)
        for i, ext_i in enumerate(ext):
            ids += [i * max_len + j for j in range(ext_i, max_len)]
        loss = self.loss_fn(y_pred.view(-1, y_pred.size(-1))[ids], y.view(-1)[ids])
        return loss
