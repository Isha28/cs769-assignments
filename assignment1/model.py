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

    input_mod = {}
    output = np.zeros((len(vocab), emb_size), dtype=np.float64)
    with open(emb_file, 'r', encoding='utf-8') as f:
        for line in f:
            line_elements = line.split()
            embedding_values = np.array(line_elements[1:], dtype=np.float64)
            input_mod[line_elements[0]] = embedding_values

    output = []
    for each_word in vocab.word2id:
        if input_mod.get(each_word) is not None:
            out = input_mod.get(each_word)
            output.append(out)
        else:
            rand_out = np.random.random(emb_size)
            output.append(rand_out)

    return output


class DanModel(BaseModel):
    def __init__(self, args, vocab, tag_size):
        super(DanModel, self).__init__(args, vocab, tag_size)
        
        self.n_vocab = len(vocab)
        self.n_embed = args.emb_size

        self.define_model_parameters()
        self.init_model_parameters()

        # Use pre-trained word embeddings if emb_file exists
        if args.emb_file is not None:
            self.copy_embedding_from_numpy()

    def define_model_parameters(self):
        """
        Define the model's parameters, e.g., embedding layer, feedforward layer.
        """
        self.embedding = torch.nn.Embedding(num_embeddings=self.n_vocab, embedding_dim=self.n_embed,padding_idx=self.vocab['<pad>'])

        self.fc1 = nn.Linear(self.n_embed, self.args.hid_size)
        self.z1 = nn.LeakyReLU(0.2)
        self.d1 = nn.Dropout(p=self.args.hid_drop)

        self.fc2 = nn.Linear(self.args.hid_size,self.args.hid_size)
        self.z2 = nn.LeakyReLU(0.2)
        self.d2 = nn.Dropout(p=self.args.hid_drop)

        self.fc3 = nn.Linear(self.args.hid_size,self.args.hid_size)
        self.z3 = nn.LeakyReLU(0.2)
        self.d3 = nn.Dropout(p=self.args.hid_drop)


        self.final = nn.Linear(self.args.hid_size,self.tag_size)


    def init_model_parameters(self):
        """
        Initialize the model's parameters by uniform sampling from a range [-v, v], e.g., v=0.08
        """
        lower_bound = -0.08
        upper_bound = 0.08
        for name, param in self.named_parameters():
            torch.nn.init.uniform_(param, lower_bound, upper_bound)


    def copy_embedding_from_numpy(self):
        """
        Load pre-trained word embeddings from numpy.array to nn.embedding
        """
        emb_file_load = load_embedding(self.vocab, self.args.emb_file, self.args.emb_size)
        self.embedding.weight = nn.Parameter(torch.tensor(emb_file_load, dtype=torch.float32))


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
        out = torch.count_nonzero(x,dim=1).to(torch.float)
        out = out.unsqueeze(-1).expand(-1,self.args.emb_size)+1e-5

        x = self.embedding(x)
        x = torch.sum(x,1)
        x = torch.div(x,out)

        if self.args.emb_drop!=0:
            x = self.embdrop(x)

        x = self.fc1(x)
        x = self.z1(x)

        x = self.d1(x)

        x = self.fc2(x)
        x = self.z2(x)

        x = self.d2(x)

        x = self.final(x)
        return x
