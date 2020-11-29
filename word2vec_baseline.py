import nltk
import re, string, spacy
nltk.download('stopwords')
from nltk.corpus import stopwords

from urllib import request
from collections import Counter
import os, zipfile
from itertools import chain

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


class BuildVocab:
    def __init__(self, dir,max_vocab_size = -1, min_counts = 200, window_size = 2):
        """
        freq (counter) : word to frequency
        vocab (dict) : word to index
        context (list) : [ [주변단어] , 중심단어]
        """
        self.dir = dir
        self.max_vocab_size = max_vocab_size
        self.min_counts = min_counts
        self.window_size = window_size
        self.stopwords = stopwords.words('english')
        self.nlp = spacy.load('en', disable=['ner', 'parser'])

        print("*** Download and Cleaning Data ***")
        self.download_dataset()
        self.cleaning_data()

        print("*** Build Vocabulary ***")
        self._get_frequency_and_vocab()

        print("*** Subsampling ***")
        self.subsampling()
        print("*** Build Context set ***")
        self.build_context_set()
    
    def download_dataset(self):

        if 'bbc-text.csv' not in os.listdir():
            filename,_ = request.urlretrieve('https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv', dir+'bbc-text.csv')
        with open('bbc-text.csv', 'r') as f:
            data = f.read()
        self.raw_data = re.split('\n\w+?,',data)[1:]

    def _clean_text(self,text):
        text = text.lower() # 소문자로 변환
        text = re.sub(r'\[.*?\]', '', text) # 대괄호에 묶인 단어 제거
        text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text) # 구두점 제거
        text = re.sub(r'\w*\d\w*', '', text) # 숫자 제거

        doc = self.nlp(text)
        text = " ".join([word.lemma_ for word in doc])
        text = text.replace('-PRON-', '')

        return [i for i in text.split() if i not in self.stopwords]

    def cleaning_data(self):
        self.data = [self._clean_text(line) for line in self.raw_data]

    def _get_frequency_and_vocab(self):
        most_common = Counter(chain.from_iterable(self.data)).most_common()
        if self.max_vocab_size > 0:
            most_common = most_common[:self.max_vocab_size]
        most_common_words = [word for word, count in most_common if count >= self.min_counts]

        self.freq = {word:0 for word in most_common_words}
    
        filtered_data = []
        for line in self.data:
            tmp = []
            for word in line:
                if self.freq.get(word) is None:
                    continue
                self.freq[word] += 1
                tmp.append(word)
            filtered_data.append(tmp)
        self.data = filtered_data

        self.freq = Counter(self.freq)
        vocab = [i for i,j in self.freq.most_common()]
        self.vocab = dict(zip(vocab, range(len(vocab)))) # 가장 많이 등장한 단어의 index가 0이 되도록

        self.vocab_size = len(self.vocab)

        print("> Vocab size :",self.vocab_size)
        print('> Most common words :', self.freq.most_common(10))

    def _get_keep_prob(self):
        total = sum(self.freq.values())
        keep_prob = lambda x: 1e-3*((1e3*x)**(1/2)+1)/x
        self.keep_rate = {word:keep_prob(freq/total) for word, freq in self.freq.items()}

    def subsampling(self):
        self._get_keep_prob()

        subsampling = []
        for line in self.data:
            tmp = []
            for word in line:
                if random.random() < self.keep_rate[word]:
                    # uniform(0,1)에서 sample하나를 뽑고 keep prob와 비교하여 keep하거나 remove
                    tmp.append(word)
            subsampling.append(tmp)
        self.data = subsampling

    def build_context_set(self): # (중심 단어, (주변 단어1, 주변 단어1)), (중심단어, (주변 단어1, 주변 단어 2, ...) 
        # self.context = []
        # for line in self.data:
        #     for i in range(len(line)):
        #         center = line[i]

        #         tmp = []
        #         for j in range(i-self.window_size, i+self.window_size+1):
        #             if j < 0 or j >= len(line) or j == i:
        #                 continue
        #             tmp.append(self.vocab[line[j]])
        #         self.context.append([tmp,self.vocab[center]])

        self.context = []
        for line in self.data:
            for i in range(len(line)):
                center = line[i]

                for j in range(i-self.window_size, i+self.window_size+1):
                    if j < 0 or j >= len(line) or j == i:
                        continue
                    self.context.append([self.vocab[line[j]],self.vocab[center]])

        print('pairs of target and context words :', len(self.context))
        print("Context examples :",self.context[10:15])

class SkipGram(nn.Module):
    def __init__(self,vocab , hidden_size, batch_size, negative_sample_size):
        super(SkipGram, self).__init__()
        self.vocab = vocab.vocab
        self.freq = vocab.freq
        self.context = vocab.context
        self.batch_size = batch_size
        self.vocab_size = vocab.vocab_size
        self.hidden_size = hidden_size
        self.negative_sample_size = negative_sample_size

        self._get_negative_sampling_prob()

        self.U = nn.Embedding(self.vocab_size, self.hidden_size)
        self.V = nn.Linear(self.hidden_size,self.vocab_size)

        self.log_sigmoid = nn.LogSigmoid()

    def dataloader(self):
        for start in range(0, len(self.context),self.batch_size):
            target_batch, center_batch = zip(*self.context[start:start+self.batch_size])

            yield center_batch, target_batch

    def _get_negative_sampling_prob(self):
        total = sum(f**(3/4) for f in self.freq.values())
        sampling_prob = lambda x: x**(3/4)/total
        self.negative_sampling_prob = torch.tensor([sampling_prob(freq) for freq in self.freq.values()])

    def _get_negative_samples(self,excluded_ids):
        temp_prob = self.negative_sampling_prob.clone()
        temp_prob[excluded_ids] = 0.

        return torch.multinomial(temp_prob, self.negative_sample_size)

    def negative_sampling(self,input_ids, target_ids):
        """
        target : 정답:1, negative sample:-1, 그외:0인 tensor
        """
        dim = len(input_ids)
        indicator = torch.zeros((dim,self.vocab_size))

        for i,(in_ids, tar_ids) in enumerate(zip(input_ids, target_ids)):
            excluded_ids = [in_ids,tar_ids]
            negative_ids = self._get_negative_samples(excluded_ids)
            
            indicator[i,negative_ids] = -1
            indicator[i,target_ids] = 1
            
        return indicator

    def forward(self,input_ids):

        center_embedding = self.U(torch.tensor(input_ids))
        output = self.V.forward(center_embedding)
        
        return output

    def loss(self,pred,indicator):
        elements = self.log_sigmoid(pred*indicator)
        loss = - elements.sum(axis = 1).mean()

        return loss

    def get_examples(self):
        self.index_to_word = {j:i for i,j in self.vocab.items()}
        # w = self.U.weight.detach()
        # sims = cosine_similarity(w)

        # for center in ['government','president','economy','minister','economic','digital','industry','foreign','campaign']:
        #     print(center, end = ' => ')
        #     i = self.vocab[center]
        #     topk = sims[i].argsort()[-8:-1][::-1]
        #     print([(index_to_word[j], sims[i][j]) for j in topk])

        w = model.U
        dist = nn.PairwiseDistance()
        for center in ['government','president','economy','minister','economic','digital','industry','foreign','campaign']:
            i = model.vocab[center]
            w_i = w(torch.tensor([i]))
            temp = []
            for j in range(model.vocab_size):
                w_j = w(torch.tensor([j]))
                temp.append([round(float(dist(w_i, w_j)),3),index_to_word[j]])
            temp.sort(key = lambda x: x[0])
            print(center, temp[1:11])
        
if __name__=='__main__':
    
    vocab = BuildVocab(dir = '' , max_vocab_size = -1 , min_counts = 50, window_size = 2)
    model = SkipGram(vocab = vocab , hidden_size = 100, batch_size = 2000, negative_sample_size = 10)

    # train
    optimizer = optim.SGD(model.parameters(), lr=0.05)

    for ep in range(20):
        i = 0
        for input_ids, target_ids in model.dataloader():
            optimizer.zero_grad()
            output = model.forward(input_ids)
            indicator = model.negative_sampling(input_ids, target_ids)
            loss = model.loss(output, indicator)
            loss.backward()
            optimizer.step()
            if i%100 == 0:
                print(loss)
            i += 1

        print(f"{ep} 끗 ***********")
        model.get_examples()
