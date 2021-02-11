import nltk
import re, string, spacy
from urllib import request
nltk.download('stopwords')
from nltk.corpus import stopwords

from collections import Counter
import os
import random
from itertools import chain

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics.pairwise import cosine_similarity


class ExampleDownloader:
    """    bbc-text를 download한 후 token화와 불용어 제거 등 기본적인 전처리를 거칩니다.    """
    def __init__(self, dir):
        self.dir = dir
        self.stopwords = stopwords.words('english')
        self.nlp = spacy.load('en', disable=['ner', 'parser'])

        print("*** Download ***")
        self.download_dataset()
        print("*** Cleaning Data ***")
        self.cleaning_data()

    def download_dataset(self):
        if 'bbc-text.csv' not in os.listdir():
            filename, _ = request.urlretrieve(
                'https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv',
                self.dir + 'bbc-text.csv')
        with open('bbc-text.csv', 'r') as f:
            data = f.read()
        self.raw_data = re.split('\n\w+?,', data)[1:]

    def _clean_text(self, text):
        text = text.lower()  # 소문자로 변환
        text = re.sub(r'\[.*?\]', '', text)  # 대괄호에 묶인 단어 제거
        text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)  # 구두점 제거
        text = re.sub(r'\w*\d\w*', '', text)  # 숫자 제거

        doc = self.nlp(text)
        text = " ".join([word.lemma_ for word in doc])
        text = text.replace('-PRON-', '')

        return [i for i in text.split() if i not in self.stopwords]

    def cleaning_data(self):
        self.raw_data = [self._clean_text(line) for line in self.raw_data]


class Vocabulary:
    """    Skip gram을 위한 vocab과 center-context set을 생성합니다.    """

    def __init__(self, data, max_vocab_size=-1, min_counts=200, window_size=2):

        """
        :param data: token화가 끝난 nested list 형태의 데이터. [[token6, token1, ...], [token5, token2, ...] ... ]
        :param max_vocab_size: vocab의 최대 크기. -1인 경우 min_counts만 고려하고, 양수인 경우 빈도 내림차순으로 정렬한 후 상위 max_vocab_size 개만을 선택합니다.
        :param min_counts: 최소 등장 횟수. min_counts 미만으로 등장한 단어는 학습에서 제외됩니다.
        :param window_size: center 좌우로 체크할 window size
        """

        self.raw_data = data
        self.max_vocab_size = max_vocab_size
        self.min_counts = min_counts
        self.window_size = window_size

        print("*** Build Vocabulary ***")
        self._get_frequency_and_vocab()

        print("*** Subsampling ***")
        self.subsampling()

        print("*** Build Context set ***")
        self.build_context_set()

    def _get_frequency_and_vocab(self):

        """    min_counts와 max_vocab_size를 고려하여 학습할 단어를 선택하고 단어의 빈도를 self.freq (counter)에 저장합니다.    """

        most_common = Counter(chain.from_iterable(self.raw_data)).most_common()
        if self.max_vocab_size > 0:
            most_common = most_common[:self.max_vocab_size]
        self.freq = {word:count for word, count in most_common if count >= self.min_counts}

        filtered_data = []
        for line in self.raw_data:
            tmp = []
            for word in line:
                if self.freq.get(word) is None:
                    continue
                tmp.append(word)
            filtered_data.append(tmp)
        self.raw_data = filtered_data

        self.freq = Counter(self.freq)
        vocab = [i for i, j in self.freq.most_common()]
        self.vocab = dict(zip(vocab, range(len(vocab))))  # 가장 많이 등장한 단어의 index가 0이 되도록

        self.vocab_size = len(self.vocab)

        print("> Vocab size :", self.vocab_size)
        print('> Most common words :', self.freq.most_common(10))

    def _get_drop_prob(self):
        """    freq를 이용해서 drop prob를 계산합니다.    """
        total = sum(self.freq.values())
        drop_prob = lambda x: max(0, 1 - (1e-4 / x) ** (1 / 2))
        self.drop_rate = {word: drop_prob(freq / total) for word, freq in self.freq.items()}

    def _drop_this_word(self, word):
        """
        uniform(0,1)에서 sample하나를 뽑고, drop prob와 비교하여 keep하거나 remove합니다.

        :param word: drop 유무를 판단할 단어
        :return: True or False (bool)
        """
        return random.uniform(0, 1) < self.drop_rate[word]

    def subsampling(self):
        """    self.data를 돌면서 self._drop_this_word(query)를 이용하여 subsampling을 진행합니다.    """
        self._get_drop_prob()

        subsampling = []
        for line in self.raw_data:
            tmp = []
            for word in line:
                if not self._drop_this_word(word):
                    tmp.append(word)
            subsampling.append(tmp)
        self.data = subsampling

    def build_context_set(self):
        """    subsampling된 결과를 이용하여 [(중심 단어1, (주변 단어1, 주변 단어2)), (중심단어2, (주변 단어1, 주변 단어 2, ...)] 를 생성합니다.    """

        self.center = []
        self.context = []

        for line in self.data:
            if len(line) < 2:
                continue
            for i in range(len(line)):  # center가 i일 때
                context = []
                center = line[i]
                for j in range(i - self.window_size, i + self.window_size + 1):
                    if j < 0 or j >= len(line) or j == i:
                        continue
                    context.append(self.vocab[line[j]])
                if context:
                    self.context.append(context)
                    self.center.append([self.vocab[center]])

        print('> pairs of target and context words :', len(self.context))
        print("> Center examples :", self.center[:5])
        print("> Context examples :", self.context[:5])

class SkipGram(nn.Module):
    def __init__(self, data, hidden_size=200, negative_sample_size=5, batch_size=200, padding_idx=0):
        """    data를 받아 vocabulary object를 생성하고, 이를 기반으로 negative sampling을 진행합니다.    """
        super(SkipGram, self).__init__()

        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.negative_sample_size = negative_sample_size
        self.padding_idx = padding_idx
        self.neg_candidates = []

        print('*** Build Vocab ***')
        self.vocab = Vocabulary(example.raw_data, max_vocab_size=-1, min_counts=30, window_size=3)
        self.vocab_size = self.vocab.vocab_size
        self.word_to_idx = self.vocab.vocab
        self.idx_to_word = {j: i for i, j in self.word_to_idx.items()}

        print('*** Negative sampling ***')
        self.negative_sampling()

        print('*** Batchify ***')
        self.batchify()

        self.U = nn.Embedding(self.vocab_size, self.hidden_size)
        self.V = nn.Embedding(self.vocab_size, self.hidden_size)

    def _cal_negative_sampling_prob(self):
        """   negative sampling을 위한 확률(여기서는 weight)을 계산합니다.   """
        sampling_weight = lambda x: x ** (0.75)
        self.negative_sampling_prob = []
        for i in range(self.vocab_size):
            word = self.idx_to_word[i]
            self.negative_sampling_prob.append(sampling_weight(self.vocab.freq[word]))
        self.negative_sampling_prob = torch.tensor(self.negative_sampling_prob)

    def _get_negative_sample(self):
        """   임의의 negative sample(index)를 return합니다. 10000개씩 미리 sampling 해놓고 pop하여 사용합니다.   """
        if len(self.neg_candidates) == 0:
            self.neg_candidates = torch.multinomial(self.negative_sampling_prob, 10000, replacement=True).tolist()
        return self.neg_candidates.pop()

    def negative_sampling(self):
        """   모든 center-context set에 대한 negative sample을 생성합니다.   """
        self._cal_negative_sampling_prob()
        self.negative_samples = []

        for i in range(len(self.vocab.center)):
            negative_ids = [self.vocab.center[i]] + self.vocab.context[i]
            remove_soon = len(negative_ids)
            while len(negative_ids) < self.negative_sample_size + remove_soon:
                neg = self._get_negative_sample()
                if neg in negative_ids:
                    continue
                negative_ids.append(neg)
            negative_ids = negative_ids[remove_soon:]
            self.negative_samples.append(negative_ids)

        print('> Negative sample :', self.negative_samples[:5])

    def batchify(self):
        """
        center-context-negative set을 batch화 시키고 이에 맞는 label과 mask를 생성합니다.
        label : context (1) negative or padding (0)
        mask : padding이 아님 (1) padding(0)
        """
        self.batch = []

        for start in range(0, len(self.vocab.center), self.batch_size):
            center_batch = self.vocab.center[start:start + self.batch_size]
            context_neg_batch = self.vocab.context[start:start + self.batch_size]
            label_batch = []
            mask_batch = []

            max_len = max(len(c) for c in context_neg_batch) + self.negative_sample_size

            for i in range(len(center_batch)):
                context_length = len(context_neg_batch[i])
                context_neg_batch[i].extend(self.negative_samples[start + i])

                padding_length = max_len - context_length - self.negative_sample_size
                context_neg_batch[i] += [self.padding_idx] * padding_length

                label = [1.] * (context_length) + [0.] * self.negative_sample_size + [0.] * padding_length
                mask = [1.] * (context_length + self.negative_sample_size) + [0.] * padding_length

                label_batch.append(label)
                mask_batch.append(mask)

            self.batch.append([torch.tensor(center_batch), torch.tensor(context_neg_batch)
                                  , torch.tensor(label_batch), torch.tensor(mask_batch)])

        print('> Batch (center)      :', self.batch[0][0].shape)
        print('> Batch (context_neg) :', self.batch[0][1].shape)
        print('> Batch (label)       :', self.batch[0][2].shape)
        print('> Batch (mask)        :', self.batch[0][3].shape)

    def forward(self, center_ids, context_neg_ids):

        center_embedding = self.U(center_ids)
        context_neg_embedding = self.V(context_neg_ids)  # (batch_size, max_len , hidden_size)

        center_embedding_t = center_embedding.transpose(1, 2)  # (batch_size, hidden_size, 1)
        dot_product = torch.bmm(context_neg_embedding, center_embedding_t)  # pairwise dot product

        return dot_product.squeeze(2)

    def get_similar_word(self, query, k = 10):

        W = self.U.weight.detach()
        x = W[self.word_to_idx[query]].unsqueeze(0)

        sims = cosine_similarity(x,W).squeeze(0)
        sims_idx = sims.argsort().tolist()

        topk = []
        for i in range(1,k+1):
            idx = sims_idx[-i-1]
            topk.append((self.idx_to_word[idx],sims[idx]))

        return topk


if __name__ == '__main__':

    torch.manual_seed(152152)
    random.seed(152152)

    example = ExampleDownloader(dir='')
    # vocab = Vocabulary(example.raw_data, max_vocab_size=-1, min_counts=30, window_size=3)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SkipGram(data = example.raw_data, hidden_size=50, negative_sample_size=5, batch_size=512, padding_idx=0)

    # train
    model.to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=0.05)

    for ep in range(10):
        i = 0
        for center_ids, context_neg_ids, label, mask in model.batch:
            optimizer.zero_grad()
            dot_product = model.forward(center_ids.to(device), context_neg_ids.to(device))

            Bceloss = nn.BCEWithLogitsLoss(mask.to(device))
            l = Bceloss(dot_product, label.to(device))

            l.backward()
            optimizer.step()

            if i % 200 == 0:
                print(l)
            i += 1

        print(f"{ep} ***********")

    # 유사한 단어 탐색
    model.to('cpu')
    model.eval()

    for center in ['digital', 'military', 'technology', 'economic']:
        print(center)
        print(model.get_similar_word(center))
        print()
