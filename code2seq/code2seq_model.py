#!/usr/bin/env python3
# Copyright (c) 2019 Technion
# Copyright (c) Facebook, Inc. and its affiliates.
# ----------------------------------------------------------------------------
# MIT License

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ----------------------------------------------------------------------------
"""
    Code2seq model is adapted from: https://github.com/tech-srl/code2seq
"""

import torch
import torch.nn as nn
import torch.nn.functional as F



class EmbeddingAttentionLayer(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.attention = torch.randn(1, dim)
        self.attention = nn.Parameter(self.attention)

    def compute_weights(self, embedded: torch.Tensor) -> torch.Tensor:
        unnormalized_weights = embedded.matmul(self.attention.t())
        attention_weights = F.softmax(unnormalized_weights, dim=1)
        return attention_weights

    def forward(self, embedded: torch.Tensor) -> torch.Tensor:
        attention_weights = self.compute_weights(embedded)
        weighted = torch.bmm(attention_weights.transpose(1, 2), embedded)
        return weighted


class Code2SeqModel(nn.Module):
    def __init__(
        self,
        token_vocab_size: int,
        subtoken_vocab_size: int,
        output_vocab_size: int,
        token_pad_idx: int,
        subtoken_pad_idx: int,
        loss_fn: nn.Module,
        n_embd: int = 128,
        rnn_dropout: float = 0.5,
        embed_dropout: float = 0.25,
    ):
        super().__init__()
        self.subtoken_embedding = nn.Embedding(
            subtoken_vocab_size, n_embd, padding_idx=subtoken_pad_idx
        )
        self.node_embedding = nn.Embedding(
            token_vocab_size, n_embd, padding_idx=token_pad_idx
        )
        self.path_lstm = nn.LSTM(
            n_embd, n_embd, bidirectional=True, dropout=rnn_dropout, batch_first=True
        )
        self.combined_layer = nn.Linear(n_embd * 4, n_embd)
        self.dropout = nn.Dropout(embed_dropout)
        self.attn_layer = EmbeddingAttentionLayer(n_embd)
        self.out_layer = nn.Linear(n_embd, output_vocab_size)

        self.loss_fn = loss_fn

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def embed_paths(self, paths):
        path_tokens_embedded = self.node_embedding(paths)
        batch_size, bag_size, path_len, _ = path_tokens_embedded.shape
        path_tokens_embedded = path_tokens_embedded.view(
            (batch_size * bag_size, path_len, -1)
        )
        out, (h, c) = self.path_lstm(path_tokens_embedded)
        paths_embedded = h.permute((1, 0, 2)).reshape(batch_size, bag_size, -1)
        return paths_embedded

    def embed_subtokens(self, subtokens):
        tokens_embedded = self.subtoken_embedding(subtokens)
        return tokens_embedded.sum(2)

    def forward(self, starts, paths, ends, targets, return_loss=False):
        # embed individual parts
        starts_embedded = self.embed_subtokens(starts)
        paths_embedded = self.embed_paths(paths)
        ends_embedded = self.embed_subtokens(ends)

        # combine by concacenating
        combined_embedded = torch.cat(
            (starts_embedded, paths_embedded, ends_embedded), dim=2
        )
        combined_embedded = self.dropout(combined_embedded)
        combined_embedded = torch.tanh(self.combined_layer(combined_embedded))

        # combine paths by simple attention
        code_embedded = self.attn_layer(combined_embedded).squeeze()
        y_pred = self.out_layer(code_embedded)

        if not return_loss:
            return y_pred
        return self.loss_fn(y_pred, targets)
