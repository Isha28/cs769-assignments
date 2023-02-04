import torch
import torch.nn as nn
import zipfile
import numpy as np
import torch.nn.functional as Fun

class BaseModel(nn.Module):
    def __init__(self, args, vocab, tag_size):
        super(BaseModel, self).__init__()
        self.args = args
        self.vocab = vocab
        self.tag_size = tag_size

    def save(self, path):
        # Save model
        print(f'Saving model to {path}')
        ckpt = {
            'args': self.args,
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }
        torch.save(ckpt, path)

    def load(self, path):
        # Load model
        print(f'Loading model from {path}')
        ckpt = torch.load(path)
        self.vocab = ckpt['vocab']
        self.args = ckpt['args']
        self.load_state_dict(ckpt['state_dict'])


def load_embedding(vocab, emb_file, emb_size):
    """
    Read embeddings for words in the vocabulary from the emb_file (e.g., GloVe, FastText).
    Args:
        vocab: (Vocab), a word vocabulary
        emb_file: (string), the path to the embdding file for loading
        emb_size: (int), the embedding size (e.g., 300, 100) depending on emb_file
    Return:
        emb: (np.array), embedding matrix of size (|vocab|, emb_size) 
    """
    input_mod = np.load(emb_file ,allow_pickle=True).item() # check
    output = []
    for each_word in vocab.word2id:
        if input_mod.get(each_word) is not None:
            out = input_mod.get(each_word)
            output.append(out)
        else:
            rand_out = np.random.random(emb_size)
            output.append(rand_out)
        
    emb = np.array(output)    
    return emb

class DanModel(BaseModel):
    def __init__(self, args, vocab, tag_size):
        super(DanModel, self).__init__(args, vocab, tag_size)
        self.define_model_parameters()
        self.init_model_parameters()

        # Use pre-trained word embeddings if emb_file exists
        if args.emb_file is not None:
            self.copy_embedding_from_numpy()

    def define_model_parameters(self):
        """
        Define the model's parameters, e.g., embedding layer, feedforward layer.
        """
        vocab_len = len(self.vocab)
        emb_size = self.args.emb_size
        iter_len = self.args.hid_layer
        hid_size = self.args.hid_size
        drop = self.args.hid_drop
        tag_size = self.tag_size
        self.embed = nn.Embedding(vocab_len, emb_size)
        ff = []
        for idx in range(iter_len):
            if idx == 0:
                ff.append(nn.Linear(emb_size, hid_size))
                ff.append(nn.Dropout(drop))
            elif idx != iter_len - 1:
                ff.append(nn.Linear(hid_size, hid_size))
                ff.append(nn.Dropout(drop))
            else:
                ff.append(nn.Linear(hid_size, tag_size))
        self.fc = nn.ModuleList(ff)

    def init_model_parameters(self):
        """
        Initialize the model's parameters by uniform sampling from a range [-v, v], e.g., v=0.08
        """
        for each_ff in self.fc:
            if type(each_ff) == nn.Linear:
                nn.init.uniform_(each_ff.weight, -0.08, 0.08)

    def copy_embedding_from_numpy(self):
        """
        Load pre-trained word embeddings from numpy.array to nn.embedding
        """
        # call function to get emb output
        vocab = self.vocab
        emb_file = self.args.emb_file
        emb_size = self.args.emb_size
        emb = load_embedding(vocab, emb_file, emb_size)
        output = torch.from_numpy(emb)
        self.embed.weight = nn.Parameter(output)
        # self.embed.weight.data.copy_(output)
        self.embed.weight.requires_grad = False

    def forward(self, x):
        """
        Compute the unnormalized scores for P(Y|X) before the softmax function.
        E.g., feature: h = f(x)
              scores: scores = w * h + b
              P(Y|X) = softmax(scores)  
        Args:
            x: (torch.LongTensor), [batch_size, seq_length]
        Return:
            scores: (torch.FloatTensor), [batch_size, ntags]
        """
        emb_out = self.embed(x)
        # emb_out = nn.Embedding(vocab_len, emb_size)
        mean_out = emb_out.mean(dim=1) # check
        for each_ff in self.fc:
            if type(each_ff) == nn.Linear:
                mean_out = Fun.relu(each_ff(mean_out))
            else:
                mean_out = each_ff(mean_out)
        return mean_out
