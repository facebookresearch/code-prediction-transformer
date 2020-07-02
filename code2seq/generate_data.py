import argparse
import json
import os
import pickle
import random
import re
from collections import defaultdict
from itertools import chain, combinations, product

from utils import get_ancestors, get_terminal_nodes, parallelize, tokenize
from tqdm import tqdm


PLACEHOLDER = "<placeholder_token>"
UNK = "<unk_token>"


def get_leaf_nodes(ast, id_type):
    # get ids for special leaf types: attr, num, name, param
    if id_type == "attr":
        types_ = {"attr"}
    elif id_type == "num":
        types_ = {"Num"}
    elif id_type == "name":
        types_ = {"NameLoad", "NameStore"}
    elif id_type == "param":
        types_ = {"NameParam"}

    nodes = []
    for i, node in enumerate(ast):
        if "type" in node and node["type"] in types_:
            nodes.append(i + 1)
    return nodes


def get_value(d):
    return d["value"] if "value" in d else d["type"]


def extract_paths(ast, max_length):
    def dfs(i):
        node = ast[i]
        if "children" not in node:
            full_paths = []
            half_paths = [[i]]
        else:
            children = node["children"]
            child_to_full_paths, child_to_half_paths = zip(
                *(dfs(child_id) for child_id in children)
            )
            full_paths = list(chain.from_iterable(child_to_full_paths))
            for i_child in range(len(children) - 1):
                for j_child in range(i_child + 1, len(children)):
                    i_child_half_paths = child_to_half_paths[i_child]
                    j_child_half_paths = child_to_half_paths[j_child]
                    for i_half_path, j_half_path in product(
                        i_child_half_paths, j_child_half_paths
                    ):
                        path_len = len(i_half_path) + len(j_half_path) + 1
                        if path_len > max_length:
                            continue
                        path = list(chain(i_half_path, [i], reversed(j_half_path)))
                        full_paths.append(path)
            half_paths = [
                half_path + [i]
                for half_path in chain.from_iterable(child_to_half_paths)
                if len(half_path) + 1 < max_length
            ]
        return full_paths, half_paths

    return dfs(0)[0]


def get_all_paths(ast, id_type, max_path_len, max_num_paths):
    if id_type == "leaves":
        nodes = get_terminal_nodes(ast)
    else:
        nodes = get_leaf_nodes(ast, id_type)
    if not nodes:
        return []
    
    all_paths = extract_paths(ast, max_path_len)
    ast_values = [get_value(i) for i in ast]
    terminal_words = [get_value(ast[i]) for i in get_terminal_nodes(ast)]
    tokenized_words = {word: tokenize(word) for word in terminal_words}
    node_to_path_idx = {i: [] for i in range(len(ast))}
    for i, path in enumerate(all_paths):
        node_to_path_idx[path[-1]].append(i)

    dps = []
    paths_to_choose_from = []
    prev_node = 0
    for node in nodes:
        for j in range(prev_node, node):
            paths_to_choose_from += [
                all_paths[path_i] for path_i in node_to_path_idx[j]
            ]
        prev_node = node

        paths_to_here = [all_paths[path_i] for path_i in node_to_path_idx[node]]
        if len(paths_to_choose_from) + len(paths_to_here) <= max_num_paths:
            paths = paths_to_choose_from.copy() + paths_to_here
        else:
            if len(paths_to_here) > max_num_paths:
                paths = random.sample(paths_to_here, max_num_paths)
            else:
                paths = paths_to_here + random.sample(
                    paths_to_choose_from, max_num_paths - len(paths_to_here)
                )

        # convert to vocab
        target = ast_values[node]
        paths = [
            [ast_values[i] if i != node else PLACEHOLDER for i in p] for p in paths
        ]
        lefts = [tokenized_words[p[0]] for p in paths]
        rights = [
            tokenized_words[p[-1]] if p[-1] != PLACEHOLDER else [PLACEHOLDER]
            for p in paths
        ]
        dps.append([target, lefts, paths, rights])
    return dps


def get_word2idx(out_fp):
    with open(out_fp, "rb") as fin:
        vocab = pickle.load(fin)
    word2idx = {word: i for i, word in enumerate(vocab)}
    word2idx = defaultdict(lambda: word2idx[UNK], word2idx)
    print("Read vocab from: {}".format(out_fp))
    return word2idx


def main():
    parser = argparse.ArgumentParser(
        description="Generate terminal to terminal paths from AST"
    )
    parser.add_argument("--ast_fp", "-a", help="Filepath with the ASTs to be parsed")
    parser.add_argument(
        "--out_fp", "-o", default="/tmp/dps.txt", help="Filepath for the output dps"
    )
    parser.add_argument("--max_path_len", type=int, default=9, help="Max path len.")
    parser.add_argument("--max_num_paths", type=int, default=200)
    parser.add_argument("--base_dir", "-b", type=str)
    parser.add_argument(
        "id_type",
        choices=["attr", "num", "name", "param", "leaves"],
        default="attr",
        help="Which ids to generate. Default = attr",
    )
    args = parser.parse_args()
    print("Max path len: {}".format(args.max_path_len))
    print("Max num paths: {}".format(args.max_num_paths))
    print("Writing to {}".format(args.out_fp))

    # read the vocabs
    base_dir = args.base_dir
    token_vocab = get_word2idx(os.path.join(base_dir, "token_vocab.pkl"))
    subtoken_vocab = get_word2idx(os.path.join(base_dir, "subtoken_vocab.pkl"))
    output_vocab = get_word2idx(os.path.join(base_dir, "output_vocab.pkl"))

    data = []
    i = 0
    c = 0
    with open(args.ast_fp, "r") as f, open(args.out_fp, "w") as fout:
        for _ in range(20):
            i += 1
            print("Starting {} / 50".format(i))
            for _ in range(5000):
                dp = json.loads(f.readline().strip())
                if len(dp) <= 1:
                    continue
                data.append(dp)
            print(" > Finished reading: {}".format(len(data)))
            for ast in tqdm(data):
                dp = get_all_paths(ast, args.id_type, args.max_path_len, args.max_num_paths)
                for target, lefts, paths, rights in dp:
                    target = output_vocab[target]
                    lefts = [[subtoken_vocab[t] for t in lst] for lst in lefts]
                    paths = [[token_vocab[t] for t in lst] for lst in paths]
                    rights = [[subtoken_vocab[t] for t in lst] for lst in rights]

                    json.dump([target, lefts, paths, rights], fout)
                    fout.write("\n")
                    c += 1
            data = []
            print(" > Finished writing to file")
    print("Wrote {} datapoints to {}".format(c, args.out_fp))


if __name__ == "__main__":
    main()
