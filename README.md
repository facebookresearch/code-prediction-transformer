This repository contains the code for the paper: Code Completion by Feeding Trees to Transformers. 

**This repo is _not_ actively maintained.**

If you use our code/implementation, please cite our paper: https://arxiv.org/abs/2003.13848


# Dataset

**generate_new_trees.py**: a preprocessing script that converts the ASTs in py150 dataset (https://www.sri.inf.ethz.ch/phog) to modified ASTs, as explained in Appendix A.1. 

**generate_vocab.py**: a script that creates the vocab corpus. In our models, we took the top 100k common vocab to be in the corpus; all other vocab is replaced by <unk_token>.

**models\/\<model\>\/generate_data.py**: the scripts to generate datasets can be found in the models directory, for all the different models (TravTrans, PathTrans, SeqTrans, SeqRNN). Note that the inputs for SeqTrans and SeqRNN are the same. This script will handle separating out the long trees into subtrees using a sliding window, as explained in Appendix A.2. In our implementation, n_context = 1000.

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

# License
Code-Prediction-Transformer is CC-BY-NC 4.0 (Attr Non-Commercial Inter.) (e.g., FAIR) licensed, as found in the LICENSE file.
