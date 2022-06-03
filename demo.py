
#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import LSTMModel
import Dataset, Vocab

import json
import os
import torch
import argparse
import logging

def predict_with_seq(seq, converted=False, ):
    rel = None
    if not converted:
        seq = vocab.convert([seq, 0]) # [data, ext] is the format expected
    seqs = [[seq, {}]]  # the {} is a mapping of attr/leaf locations, not needed here
    batch = Dataset.collate(seqs, pad_idx)
    x = batch["input_seq"]
    y = batch["target_seq"]
    ext = batch["extended"]
    y_pred = model(x, y, ext, rel=rel, return_loss=False)
    return y_pred.squeeze()

def get_top_pred(pred, k=10, print_results=True):
    softmax = torch.nn.Softmax()
    top_perc, top_idx = torch.topk(softmax(pred), k)
    top_perc = top_perc.tolist()
    top_tokens = [vocab.idx2vocab[i] for i in top_idx]

    if print_results:
        print('Top {} predictions:'.format(k))
        for i, (perc, token) in enumerate(zip(top_perc, top_tokens)):
            print('{}) {:<12} ({:.2f}%)'.format(i + 1, token, 100 * perc))
    return top_perc, top_tokens

def predict_next(input_seq, k=10, print_results=False):
    y_pred = predict_with_seq(input_seq + ['<pad_token>'])
    top_perc, top_tokens = get_top_pred(y_pred[-1], k, print_results)
    return top_perc, top_tokens


def demo_sequence(input_seq):
    print(' '.join(input_seq))
    top_perc, top_tokens = predict_next(input_seq, print_results=True)

def demo_datapoint(data, dp_raw, idxs, converted=False, print_results=True):
    k = 10
    # predict for the whole sequence in one shot
    y_pred = predict_with_seq(data, converted)
    for i in idxs:
        context = dp_raw[max(0, i-5): i]
        target = dp_raw[i]
        print('Context: {}'.format('<before>...' + ' '.join(context)))
        print('Target : {}'.format(target))
        top_perc, top_tokens = get_top_pred(y_pred[i-1], k, print_results)
        rank = top_tokens.index(target) if target in top_tokens else -2
        print('Rank   : {}'.format(rank + 1))
        print()

def parse_args():
    parser = argparse.ArgumentParser(description="Demo for a trained model")
    parser.add_argument("--base_dir", "-b", default="/tmp/gpt2")
    parser.add_argument("--model_fp", "-m", default="rnn.pth", help="Relative fp to best_model")
    parser.add_argument("--vocab_fp", "-v", default="vocab.pkl", help="Relative fp to vocab pkl")
    parser.add_argument("--dps_fp", help="Test filepath with raw data points")
    parser.add_argument("--conv_fp", help="Test filepath with converted data points")
    parser.add_argument(
        "--ids_fp", help="Filepath with the ids that describe locations of various attrs/leaf/etc"
    )
    args = parser.parse_args()
    logging.info("Base dir: {}".format(args.base_dir))
    return args

def main():

    global vocab
    global model
    global pad_idx

    args = parse_args()
    base_dir = args.base_dir
    model_fp = os.path.join(base_dir, args.model_fp)
    vocab = Vocab(os.path.join(base_dir, args.vocab_fp))
    pad_idx = vocab.pad_idx
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)
    n_ctx=100

    model = LSTMModel(
        vocab_size=len(vocab),
        n_embd=300,
        loss_fn=loss_fn,
        n_ctx=n_ctx,
    )
    print('Created {} model!'.format(model_fp))

    # load model
    new_checkpoint = {}
    checkpoint = torch.load(model_fp, map_location=torch.device('cpu'))
    for name, weights in checkpoint.items():
        name = name.replace('module.', '')
        new_checkpoint[name] = weights
    del checkpoint

    model.load_state_dict(new_checkpoint)
    model.eval()

    print('Loaded model from:', model_fp)


    # 1. Try prediction with some made up sequence
    input_seq = ['with', 'open', '(', 'raw_fp', ',', '"r"', ')', 'as', 'fin', ':', 'data_raw', '=', '[', 'json', '.', ]
    demo_sequence(input_seq)
    demo_sequence(input_seq + ['loads'])

    # 2. Prediction on a sample from our dataset

    # read dataset
    if (args.dps_fp is not None):
        raw_fp = os.path.join(base_dir, args.dps_fp)
        with open(raw_fp, 'r') as fin:
            data_raw = [json.loads(line) for line in fin.readlines()]
        print('Read {} datapoints!'.format(len(data_raw)))
        # TODO make these random
        dp_i = 231
        idx = 50
        print('Raw data point [data, ext] = ', data_raw[dp_i])
        dp_raw = data_raw[dp_i][0]  # data_raw[dp_i][1] is an ext, we don't need it
        demo_datapoint(dp_raw, dp_raw, {idx}, converted=False)
    else:
        return

    # we can also predict from pred-converted data points
    if (args.conv_fp is not None):
        conv_fp = os.path.join(base_dir, args.conv_fp)
        with open(conv_fp, 'r') as fin:
            data_conv = [json.loads(line) for line in fin.readlines()]
        print('Converted data point [data, ext] = ', data_conv[dp_i])
        demo_datapoint(data_conv[dp_i], dp_raw, {idx}, converted=True)

    # let's focus on the attrs in this data point
    if (args.ids_fp is not None):
        ids_fp = os.path.join(base_dir, args.ids_fp)
        with open(ids_fp, 'r') as fin:
            data_ids = [json.loads(line) for line in fin.readlines()]
        print('Datapoint:\n{} .... <continued>'.format(' '.join(dp_raw[:100])))
        print('# of value predictions:')
        for name, lst in data_ids[dp_i].items():
            print('{}: {}'.format(name, len(lst)))
        attrs = data_ids[dp_i]["attr_ids"]
        demo_datapoint(dp_raw, dp_raw, attrs, converted=False, print_results=False)

        
if __name__ == "__main__":
    main()
