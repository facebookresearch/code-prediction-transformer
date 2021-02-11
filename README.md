This repository contains the code for the paper: Code Completion by Feeding Trees to Transformers. 

**This repo is _not_ actively maintained.**

If you use our code/implementation, please cite our paper: https://arxiv.org/abs/2003.13848


# Dataset

**generate_new_trees.py**: a preprocessing script that converts the ASTs in py150 dataset (https://www.sri.inf.ethz.ch/phog) to modified ASTs, explained further below in **Modifying the AST**.

**generate_vocab.py**: a script that creates the vocab corpus. In our models, we took the top 100k common vocab to be in the corpus; all other vocab is replaced by <unk_token>.

**models\/\<model\>\/generate_data.py**: the scripts to generate datasets can be found in the models directory, for all the different models (TravTrans, PathTrans, SeqTrans, SeqRNN). Note that the inputs for SeqTrans and SeqRNN are the same. This script will handle separating out the long trees into subtrees using a sliding window, as explained below in **Splitting Large Trees**. In our implementation, n_context = 1000.

**models\/trav_trans\/generate_ast_ids.py**: a script to generate index locations for certain predictions in the dataset for the AST-based models (TravTrans). Can get indices for:
- specific values (Table 9): predicting indices for attribute access, name (varialbe, module), numeric constant, function parameter name. 
- specific types (Table 10): predicting indices for function call, assign, return, list, dictionary, raise.
- All values / types (Table 6): predicting indices for all values (leaf nodes) and all types (internal nodes).

**models\/seq\/generate_data.py**: a script that is also used to generate index locations for certain predictions in the dataset for the source code-based models (SeqTrans, SeqRNN). Can get indices for:
- specific values (Table 9): same definition as above
- All values (Table 6): predicting indices for all equivalent leaf nodes of the AST (equivalent to All values, as explained above).

**dataset.py**: contains class objects (BaseSetup, BaseVocab, BaseDataset) for setting up the dataset and vocab for the model. Each model inherits from these class objects to fit the model's specific setup requirements. The BaseDataset object also contains the collate function for processing batches.

# Model
**model.py** contains the models used in this paper. SeqRNN uses **LSTMModel**, and the other models 
except Code2seq use **TransformerModel**. We used the following hyperparameters for our implementation:
- n_layer = 6
- n_embd = 300
- n_head = 6
- layer_norm_epsilon = 1e-6,
- lr = 1e-3

**code2seq/code2seq_model.py** contains the code2seq PyTorch adaption. We used the same hyperparameters as the one
from the original work, except changing the number of vocab to 100k to stay consistent with the other
models used in the paper.

# Data processing Information

## Modifying the AST
For the AST, we want the internal AST nodes to only have type information, and the leaf nodes to have value information. This way, our model can predict one information given a node (instead of both type and value). However, in the py150 dataset, there are internal and leaf nodes with both type and value information. To accomodate for this, we slightly modify the trees to fit our definition of ASTs. For nodes with both type and value information, we take the value information, and create a new node (now a leaf node) as the node's first child. Figure below illustrates an example of the modification. This increases the average number of nodes in a tree from 623.4 to 951.9.

 Before:


                  Type:AttributeLoad
                           0
                  /                  \
                 /                    \
        Type: NameLoad             Type: Attr
        Value: logging            Value: getLogger
              1                         2


After:


                  Type:AttributeLoad
                           0
                  /                  \
                 /                    \
        Type: NameLoad            Type: Attr
              1                       3
              |                       |
              |                       |
       Value: logging           Value: getLogger
              2                       4


## Splitting Large Trees
For neural network models, we need to set a maximum number of nodes in the tree that the model can take as input. Ideally, we would want to set the maximum to be high enough to take in any tree of any length; however, in practice, this is infeasible due to memory constraints (and the number of nodes could be infinitely large hypothetically.) We choose the maximum number of context (number of nodes) to be 1000, inspired by the maximum number of context set by GPT2 models and as this covers > 70% of the training data. For trees with number of nodes greater than 1000, we deploy a technique adopted by [1]. Given a large tree, we slice it into shorter segments with a sliding window (in our implementation, we used 500, which is half the context). For example, if a tree has 1700 nodes, we would have 3 new shorter trees: from nodes 0-999, nodes 500-1499, and 699-1699. For the last two trees, we would take loss and evaluate only on the nodes that the model has not seen before (1000-1499 and 1500-1699, respectively). In this way, we provide each subsequent shorter segment with some previous context, while increasing the number of training and testing datapoints at a reasonable amount (in our datasets, it doubled the number). An improvement to this sliding window technique would be to maintain the hidden states at each segment to pass along more context information, as explained in [2]. 


[1] [Character-Level Language Modeling with Deeper Self-Attention](https://arxiv.org/abs/1808.04444): Rami Al-Rfou, Dokook Choe, Noah Constant, Mandy Guo, Llion Jones. 

[2] [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/abs/1901.02860): Zihang Dai, Zhilin Yang, Yiming Yang, Jaime Carbonell, Quoc V. Le, Ruslan Salakhutdinov


# License
Code-Prediction-Transformer is CC-BY-NC 4.0 (Attr Non-Commercial Inter.) (e.g., FAIR) licensed, as found in the LICENSE file.
