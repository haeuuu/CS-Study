from urllib import request
from collections import Counter
import zipfile
import os
import torch
import torch.nn as nn
import torch.optim as optim

class BuildVocab:
    def __init__(self, dir, vocab_size, limit = -1, window_size = 2):
        """
        freq (counter) : word to frequency
        vocab (dict) : word to index
        context (list) : [ [주변단어] , 중심단어]
        """
        self.dir = dir
        self.vocab_size = vocab_size
        self.limit = limit
        self.window_size = window_size

        print("*** Download Data ***")
        self.download_dataset()

        print("*** Build Vocabulary ***")
        self.build_vocab()

        print("*** Build Context set ***")
        self.build_context_set()

    def download_dataset(self):

        if 'text8.zip' not in os.listdir():

            filename, _ = request.urlretrieve('http://mattmahoney.net/dc/text8.zip', 'text8.zip')

            with zipfile.ZipFile(filename) as z:
                z.extract(self.dir+'text8', self.dir)

        with open(self.dir + 'text8', 'r') as f:
            self.data = f.read().split()

        if self.limit > 0:
            self.data = self.data[:self.limit]

        print("> Data :",len(self.data))

    def build_vocab(self):
        most_common = Counter(self.data).most_common(self.vocab_size-1)
        most_common.append(('UNK',0))

        self.freq = Counter(dict(most_common))
    
        for i in range(len(self.data)):
            if self.freq.get(self.data[i]) is None:
                self.data[i] = 'UNK'
                self.freq['UNK'] += 1

        vocab = [i for i,j in self.freq.most_common()]
        self.vocab = dict(zip(vocab, range(len(vocab)))) # 가장 많이 등장한 단어의 index가 0이 되도록

        print("> Vocab size :",len(self.vocab))
        print('> Most common words :', self.freq.most_common(10))

    def build_context_set(self): # (중심 단어, (주변 단어1, 주변 단어1)), (중심단어, (주변 단어1, 주변 단어 2, ...) 의 index
        self.context = []
        for i in range(len(self.data)):
            center = self.data[i]
            if center == 'UNK':
                continue
                
            temp = []
            for j in range(i-self.window_size, i+self.window_size+1):
                if j == i or j < 0 or j >= len(self.data) or self.data[j] == 'UNK':
                    continue
                temp.append(self.vocab[self.data[j]])
            self.context.append([temp,self.vocab[center]])

        print('length :', len(self.context))
        print("Context examples :",self.context[10:15])
        

class SkipGram(nn.Module):
    def __init__(self,vocab , hidden_size, batch_size):
        super(SkipGram, self).__init__()
        self.vocab = vocab.vocab
        self.context = vocab.context
        self.batch_size = batch_size
        self.vocab_size = vocab.vocab_size
        self.hidden_size = hidden_size

        self.U = nn.Linear(self.vocab_size, self.hidden_size)
        self.V = nn.Linear(self.hidden_size, self.vocab_size)

        self.softmax = nn.Softmax(1)

    def dataloader(self):
        for start in range(0, len(self.context)+self.batch_size,self.batch_size):
            words_batch, center_batch = zip(*self.context[start:start+self.batch_size])
            
            input = torch.zeros((self.batch_size,self.vocab_size))
            input[range(self.batch_size),center_batch] = 1.

            target = torch.zeros((self.batch_size,self.vocab_size))
            for i in range(self.batch_size):
                target[i,words_batch[i]] = 1

            yield input, target

    def forward(self,input):

        hidden = self.U.forward(input)
        output = self.V.forward(hidden)

        prob = self.softmax(output)
        
        return prob

    def loss(self,pred,answer):

        log_prob = torch.log(pred)
        mul = log_prob*answer
        loss_per_point = -mul.sum(axis = 1)
        loss = loss_per_point.mean()
        
        return loss
        
        
        
if __name__=='__main__':
    dir = ''
    vocab_size = 10000
    limit = -1
    window_size = 2

    vocab = BuildVocab(dir,vocab_size, limit, window_size)
    model = SkipGram(vocab, 100,10000)

    # train
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    for ep in range(3):
        for i, (input, target) in enumerate(model.dataloader()):
            optimizer.zero_grad()
            prob = model.forward(input)
            loss = model.loss(prob, target)
            loss.backward()
            optimizer.step()
            if i%20 == 0:
                print(loss)
        print('ep :', loss)
