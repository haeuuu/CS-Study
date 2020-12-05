# Python으로 구현하기

#### 여정

아이디어는 이해가 되는데 디테일한 부분에는 여전히 의문이 있었다. (nn.Linear로 bias까지 학습해야 하는지, softmax의 미니 버전이 맞는지 혹은 paper에서 찾은 loss 식이 맞는지, sub/neg sampling은 epoch마다 하는건지 등...)

다른 사람들의 구현을 보고 헷갈리는 부분을 잡아야겠다 했는데 ... 일단 한글 자료에서는 못찾았다 ㅎㅎ 대부분 baseline만 구현하셨다. 영어로된 자료도 CBOW만 나왔다 ㅜㅜ 내가 헷갈리는 부분은 다 skip gram에서 어떻게 적용되는지인데 !

일단 바닥부터 다 짜놓고 학습이 잘 되고 있나 ~ 확인해봤는데 loss는 줄고있지만 실제로 단어는 제대로 맞추지 못하고 있었다.

epoch을 50번 돌려도 초기 weight 값에 영향을 크게 받는거보니 어디선가 잘못되었구나 싶었다.



다음 두가지를 참고하면서 고쳐보고 구현시에 부딫힌 문제와 고민들을 구구절절 적어보았다.



1. gensim의 word2vec 패키지를 까본다.

   > gensim은 속도 개선을 위해 **cython으로 구현**된 부분이 있는데 c의 기초만 공부했어도 **눈치껏 이해할 수 있다**. (==나) 걱정 말자!

   * loss는 어떻게 계산했는가?
   * negative sampling은?

2. mxnet으로 된 강의자료를 참고한다.



# Negative sampling 빠르게 하기

확률값이 주어졌을 때 sampling 함수는 어떤 다양한 방법을 짤 수 있을까? 뭐가 가장 빠를까? 에 대한 고찰 ...



처음에 짤 때는 아 이건 다항분포다 ! 하며 `torch.multinomial`로 구현했다.

분포라는 단어에 꽂혀버려서 복원? 비복원? 을 고민했지 multinomial을 안써도 되겠다는 생각을 전혀 하지 못했는데, gensim 패키지의 쏘 씸쁠한 구현을 보고 반성했다. ㅠㅠ 참고한 mxnet 강의자료에서는`random.chioce(weight = freq)`로 뽑았다.



## 1 ) distribution을 이용하자.

나는 처음에 negative sampling을 data마다 5개씩 하도록 짰다. negative sampling을 이해하면서 자연스럽게 '매 데이터마다 5~20개씩 뽑으면 되겠군' 생각했다. `torch.multinomial`을 이용했다.

* 복원? 비복원? => 비복원 추출을 하자.
  * 왜? 굳이 중복된 값을 뽑을 필요가 없다. 우연의 일치로 5개가 모두 같은 단어가 나왔다면, 5개를 sampling한 의미가 없다. 하나의 정보만 가져가게 된다.

문제점이라고 한다면 ... 시간이 꽤 걸린다는 점이다.



다른 구현을 참고해보니 복원 추출 + **미리 만 개를 뽑아놓은 후 5개씩 잘라서 쓰고 있었다.**

```python
random.choices(population, sampling_weights, k=10000)
```

이래도 되나 ... 고민을 잠시 해보니 이래도 된다! 는 결론이 났다 ㅎㅎㅋㅋ

두 방법 모두 같은 기댓값을 가지며 n이 충분히 크다면 sample에 의한 관측값이 잘 근사할거니까.



##### [확인해보자! ]임의의 단어에 대해 기댓값이 같을까?

(k+1)개의 단어가 있고 각 단어가 뽑힐 확률 pi가 주어져있다.

이를 이용해서 5000개의 sample을 구성하려고 한다. 다음 두 방법에 대해 x~k+1~ 번째 단어가 등장하는 빈도의 기댓값은 다를까?

1. 5개씩 sampling하는 시행을 1000번 한다. (각 시행은 독립이다.)

   * 상황 : multinomial(n = 5, (p~1~ , ... , p~k+1~ ), 비복원) 을 독립적으로 1000번 한다.

     Y ; 1000번 시행시 X~k+1~ 의 갯수, X = 한 번 시행시 X~k+1~ 의 갯수라면 Y = 1000X

     E(Y)를 구하자.

   * E(X~k+1~) , P(X~k+1~)는? 

     나머지 단어가 어떻게 뽑히든지 **우리가 원하는 x~k+1~ 만 x개가 뽑히면 된다 !**

     결국 X~k+1~ v.s. 나머지 단어가 되고, 비복원 추출이므로 초기하 분포를 따른다. **즉 E(X~k+1~) = 5p~k+1~** 

   * **E(Y) = 1000*E(X) = 5000p~k+1~** 

2. 미리 2500개씩 뽑아놓고 차례로 5개씩 분배한다.

   * 상황 : multinomial(n = 2500, (p~1~ , ... , p~k+1~ ), 복원) 을 독립적으로 2번 한다.

     Y ; 2번 시행시 X~k+1~ 의 갯수, X = 한 번 시행시 X~k+1~ 의 갯수라면 Y = 2X

   * E(X~k+1~) , P(X~k+1~)는? 

     역시 나머지 단어가 어떻게 뽑히든지 **우리가 원하는 x~k+1~ 만 x개가 뽑히면 된다 !**

     복원 추출이므로 이항 분포를 따른다. **즉 E(X~k+1~) = 2500p~k+1~**

   * **E(Y) = 2*E(X) = 5000p~k+1~** 



기댓값이 같다. 즉 n이 충분히 크다면 sampling 결과는 비슷할 것으로 예상할 수 있다.

차이점은 ""분포를 얼마나 잘 재현해낼 수 있는가""가 되겠다.

주사위를 10번 던져서 만든 table과 10000번 던져 만든 table이 다른 것처럼 ...

전자는 1000번 실험했으니 두 번만 실험한 후자보다 원래 기댓값과 유사한 분포를 보여줄 것이다, 실험해보자 !



 vocab size = 10000으로 가정하고 먼저 임의로 만 개의 확률값을 생성한다.

```python
# 임의로 100개의 확률값을 생성

weight = [random.randint(0,1000) for _ in range(100)]
prob = [w/sum(weight) for w in weight]

# to torch tensor
prob = torch.tensor(prob)
```



data는 10000개라고 가정하자.

```python
print('방법 1. 비복원 추출로 5개씩 10000번')
start = time.time()
res1 = []
for _ in range(10000):
    res1.extend(torch.multinomial(prob, 5, replacement = False).tolist())
print(f">> {time.time()-start:.3f}")

print('방법 2. 복원 추출로 5000개씩 10번')
start = time.time()
res2 = []
for _ in range(10):
    res2.extend(torch.multinomial(prob, 5000, replacement = True).tolist())
print(f">> {time.time()-start:.3f}")
```



먼저 수행 시간부터 비교해보자. 100배 이상 차이가 난다.

```
방법 1. 비복원 추출로 5개씩 10000번
>> 0.707
방법 2. 복원 추출로 5000개씩 10번
>> 0.006
```



sample이 실제 분포를 얼마나 잘 재현해주는지 확인해보자. 둘 다 다행히 기댓값에 잘 근사하고 있다.

![image-20201202172320932](../fig/image-20201202172320932.png)





## 2 ) gensim 패키지는 어떻게 구현했을까?

#### `binary_search`를 이용했다 !



과녁의 넓이를 확률로 해석하던 중학교 시절의 simple한 아이디어를 이용하면 binary search를 이용해서 sampling을 쉽게할 수 있다 !

![image-20201202050029187](../fig/image-20201202050029187.png)

1. 먼저 freq를 이용해서 각 단어에 대한 확률을 계산한다.

2. 이를 `cum_table`이라는 1차원 array에 '**누적으로**' 저장한다.

   0~1의 수직선 위에서, 확률이 큰 단어는 더 많은 면적을 할당받는다.

3. 이제 0~1 사이에서 random값 하나를 뽑는다. ( `(현재 random값>>16)%전체 단어수`  )

4. 주어진 값이 어떤 단어의 영역에 들어가있는지 binary search를 통해 탐색한다.

5. 다음 random값을 `(이전 random값 * MAX + 11)%MAX`로 할당한다.

   > gensim에서 MAX = 281474976710655ULL로 되어있는데 일단 적당히 아주 큰 수로 생각했다.
   > (검색해도 안나온다 ㅜㅜ 더 찾아봐야징)
   >
   > 11은 데이터에서 나온 값이 아닌걸로 봐서 다음 random값 생성을 위해 임의의 수를 넣은것같다.
   >
   > 나는 uniform(0,1)에서 추출했다.



이건 방법만 놓고 보면 multinomial(n=1, 복원) 을 50000번 하는 것과 같다는 생각 !

어떻게 구현했느냐의 차이인거같다.

당연히  multinomial(n=1, 복원)을 50000번 돌린 것보다 훠어엉어어엉ㅇ어얼씬 빠르다.



코드는 cython으로 짜여져있는데 나처럼 C 의 기초 문법만 알지라도 ... 전체 로직이 어떻게 굴러가는지는 이해할 수 있다 !



#### Data 전처리 시에 누적 table을 만들어놓는다. [gensim/models/word2vec.py](https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/models/word2vec.py)

```python
    def make_cum_table(self, domain=2**31 - 1):
        """Create a cumulative-distribution table using stored vocabulary word counts for
        drawing random words in the negative-sampling training routines.
        To draw a word index, choose a random integer up to the maximum value in the table (cum_table[-1]),
        then finding that integer's sorted insertion point (as if by `bisect_left` or `ndarray.searchsorted()`).
        That insertion point is the drawn index, coming up in proportion equal to the increment at that slot.
        """
        vocab_size = len(self.wv.index_to_key)
        self.cum_table = np.zeros(vocab_size, dtype=np.uint32)

        # sum(f**0.75 for f in word_freq)를 위한 과정
        
        train_words_pow = 0.0
        for word_index in range(vocab_size):
            count = self.wv.get_vecattr(word_index, 'count')
            train_words_pow += count**self.ns_exponent
            
        # 누적 확률값 계산
        
        cumulative = 0.0
        for word_index in range(vocab_size):
            count = self.wv.get_vecattr(word_index, 'count')
            cumulative += count**self.ns_exponent
            
            # 0.000xxxx처럼 아주 작은 확률값 cum/train_pow에 domain을 곱해줘서 크기를 키운다.
            self.cum_table[word_index] = round(cumulative / train_words_pow * domain)
            
        if len(self.cum_table) > 0:
            
            # cummulative/train_words_pow는 마지막에 1이 되므로 assert 문으로 체크
            assert self.cum_table[-1] == domain
```



####   [gensim/models/word2vec_inner.pyx](https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/models/word2vec_inner.pyx)

```c
cdef unsigned long long w2v_fast_sentence_sg_neg(
    const int negative, np.uint32_t *cum_table, unsigned long long cum_table_len,
    REAL_t *syn0, REAL_t *syn1neg, const int size, const np.uint32_t word_index,
    const np.uint32_t word2_index, const REAL_t alpha, REAL_t *work,
    unsigned long long next_random, REAL_t *words_lockf,
    const np.uint32_t lockf_len, const int _compute_loss, REAL_t *_running_training_loss_param) nogil:

    cdef long long a
    cdef long long row1 = word2_index * size, row2
    cdef unsigned long long modulo = 281474976710655ULL
    cdef REAL_t f, g, label, f_dot, log_e_f_dot
    cdef np.uint32_t target_index
    cdef int d

    memset(work, 0, size * cython.sizeof(REAL_t))

    for d in range(negative+1): /* context 1개와 negative개 만큼 돌면서 진행*/
        if d == 0:
            target_index = word_index
            label = ONEF
        else:
			/*binary search를 이용해서 negative sample 한개 추출*/
            target_index = bisect_left(cum_table, (next_random >> 16) % cum_table[cum_table_len-1], 0, cum_table_len)
                
            /*다음 random값 미리 계산*/
            next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo
            if target_index == word_index:
                continue

	...생략...
                    
    return next_random
```

```python
"""Train on a single effective word from the current batch, using the Skip-Gram model.
    In this model we are using a given word to predict a context word (a word that is
    close to the one we are using as training). Negative sampling is used to speed-up
    training.
    Parameters
    ----------
    negative
        Number of negative words to be sampled.
    cum_table
        Cumulative-distribution table using stored vocabulary word counts for
        drawing random words (with a negative label).
    cum_table_len
        Length of the `cum_table`
    syn0
        Embeddings for the words in the vocabulary (`model.wv.vectors`)
    syn1neg
        Weights of the hidden layer in the model's trainable neural network.
    size
        Length of the embeddings.
    word_index
        Index of the current training word in the vocabulary.
    word2_index
        Index of the context word in the vocabulary.
    alpha
        Learning rate.
    work
        Private working memory for each worker.
    next_random
        Seed to produce the index for the next word to be randomly sampled.
    words_lockf
        Lock factors for each word. A value of 0 will block training.
    _compute_loss
        Whether or not the loss should be computed at this step.
    _running_training_loss_param
        Running loss, used to debug or inspect how training progresses.
    Returns
    -------
    Seed to draw the training word for the next iteration of the same routine.
    """
```



마지막으로 세 방법을 한꺼번에 비교해보았다.

```python
print('방법 3. uniform + binary search로 50000번')
start = time.time()
res3 = []
for _ in range(50000):
    next_random = random.uniform(0,1)
    res3.append(bisect.bisect_left(cum_prob, next_random))
print(f">> {time.time()-start:.3f}")
```

```python
방법 1. 비복원 추출로 5개씩 10000번
>> 0.707
방법 2. 복원 추출로 5000개씩 10번
>> 0.006
방법 3. binary search로 50000번
>> 0.037
```



세 방법 모두 기댓값에 근사하게 추출되었다 !

![image-20201202172203500](../fig/image-20201202172203500.png)



# Loss 계산하기

`word2_index` : 중심 단어

`word_index` : 현재 뽑은 단어(context일수도, negative일수도 있다.)

negative+1 ( 1은 context 한개를 의미 )만큼 돌면서

1. context 또는 negative를 뽑는다.
2. 중심 단어의 embedding과 현재 단어인 embedding을 내적한다.
3. 내적 결과를 EXP_TABLE에서 얻어온다. `f`
4. context일 경우는 그냥 내적값을, neg의 경우에는 -1을 곱해준다.
5. log를 거친다. ( 내가 이해한 식 그대로 잘 따라감 ! )



torch.bmm => pair wise dot product. 내적의 batch ver !



정의한 식을 maximize하는거니까 구현시에는 -1을 곱해서 minimize하도록 해야한다.



#### 처음에 학습이 안되었던 이유

mask 안하고 label으로 다 해결하려다가 틀렸다!

dot_product에는 padding에 대한 내적값도 포함되어있으므로 얘를 버려줘야 한다.

처음에는 label을 1(context), -1(negative), 0(padding)으로 주고

dot_product*label을 이용해서 padding에 대한 내적값을 없애려고 했다.

그런데 생각해보니, 내적 후에 sigmiod => log를 취해야 최종 loss가 되는데, 내적=0이면 없어지는게 아니라 log(sigmoid(x = 0)) = log(0.5) 로 loss가 발생한다 !!!!

즉 내적에서 없애는게 아니라 log까지 취한 후 0을 곱해야하는데 잘못계산하고 있었다.



reshape을 사용했음

=> 그래서 batch size에 따라 학습 결과가 아주 크게 달랐구나... batch가 클 수록 데이터들이 더 섞여서 엉뚱하게 학습됨.





```python
cdef unsigned long long w2v_fast_sentence_sg_neg(
    const int negative, np.uint32_t *cum_table, unsigned long long cum_table_len,
    REAL_t *syn0, REAL_t *syn1neg, const int size, const np.uint32_t word_index,
    const np.uint32_t word2_index, const REAL_t alpha, REAL_t *work,
    unsigned long long next_random, REAL_t *words_lockf,
    const np.uint32_t lockf_len, const int _compute_loss, REAL_t *_running_training_loss_param) nogil:
    """Train on a single effective word from the current batch, using the Skip-Gram model.
    In this model we are using a given word to predict a context word (a word that is
    close to the one we are using as training). Negative sampling is used to speed-up
    training.
    Parameters
    ----------
    negative
        Number of negative words to be sampled.
    cum_table
        Cumulative-distribution table using stored vocabulary word counts for
        drawing random words (with a negative label).
    cum_table_len
        Length of the `cum_table`
    syn0
        Embeddings for the words in the vocabulary (`model.wv.vectors`)
    syn1neg
        Weights of the hidden layer in the model's trainable neural network.
    size
        Length of the embeddings.
    word_index
        Index of the current training word in the vocabulary.
    word2_index
        Index of the context word in the vocabulary.
    alpha
        Learning rate.
    work
        Private working memory for each worker.
    next_random
        Seed to produce the index for the next word to be randomly sampled.
    words_lockf
        Lock factors for each word. A value of 0 will block training.
    _compute_loss
        Whether or not the loss should be computed at this step.
    _running_training_loss_param
        Running loss, used to debug or inspect how training progresses.
    Returns
    -------
    Seed to draw the training word for the next iteration of the same routine.
    """
    cdef long long a
    cdef long long row1 = word2_index * size, row2
    cdef unsigned long long modulo = 281474976710655ULL
    cdef REAL_t f, g, label, f_dot, log_e_f_dot
    cdef np.uint32_t target_index
    cdef int d

    memset(work, 0, size * cython.sizeof(REAL_t))

    for d in range(negative+1):
        if d == 0:
            target_index = word_index
            label = ONEF
        else:
            target_index = bisect_left(cum_table, (next_random >> 16) % cum_table[cum_table_len-1], 0, cum_table_len)
            next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo
            if target_index == word_index:
                continue
            label = <REAL_t>0.0

        row2 = target_index * size
        f_dot = our_dot(&size, &syn0[row1], &ONE, &syn1neg[row2], &ONE)
        if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (label - f) * alpha

        if _compute_loss == 1:
            f_dot = (f_dot if d == 0  else -f_dot)
            if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
                continue
            log_e_f_dot = LOG_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
            _running_training_loss_param[0] = _running_training_loss_param[0] - log_e_f_dot

        our_saxpy(&size, &g, &syn1neg[row2], &ONE, work, &ONE)
        our_saxpy(&size, &g, &syn0[row1], &ONE, &syn1neg[row2], &ONE)

    our_saxpy(&size, &words_lockf[word2_index % lockf_len], work, &ONE, &syn0[row1], &ONE)

    return next_random

```



# Batch 만들 때 padding하기

embedding을 indexing하는 과정에서 `context+negative`의 길이가 제각각이면 안된다 !

0으로 padding한 후에, padding인지 아닌지를 나타내주는 label을 만들어놓고 loss 계산시에는 뺄 수 있도록 해주자.





# DO NOT USE `.reshape, .view`

[reshape과 view로 인해 학습이 제대로 진행되지 않는다.](https://discuss.pytorch.org/t/for-beginners-do-not-use-view-or-reshape-to-swap-dimensions-of-tensors/75524)



왠지... mxnet은 epoch 5로도 충분했는데

내껀 계속 학습이 전~혀 안돼고 있었다.

loss에서 mask 유형이 세개라 그런가? 0을 곱해서 해당 값을 무시하도록 만든게 중간에 꼬여버렸나? 싶었는데

코드를 다시 천천히 읽어보니 왠지 reshape이 찝찝했다.

구글 박사님께 질문을 하니 위와같은 글을 발견할 수 있었다.



`transpose` 또는 `permute`를 쓰자.







# torch.sigmoid v.s. nn.Sigmoid

> function v.s. Module

[출처](https://stackoverflow.com/questions/55621322/why-is-torch-nn-sigmoid-a-class-instead-of-a-method)

둘은 똑같다.

그저 후자는 nn.Sequential 안에 넣을 때 편하다.

torch는 lua torch7 based로 만들어졌는데 여기서 모든 미분 가능한 nn 함수가 모듈로 수행되기 때문에 모듈이 존재하는거같다.







# BCEwithLogitLoss

* Sigmoid를 거친 후 BCELoss에 넣는 것을 하나로 합친 것.

* log-sum-exp trick 을 통해 numerical stability를 가짐.

  * 너무 큰 수나 작은 수의 연산을 안정적으로 하기 위해(overflow 없이) exp와 log를 이용해서 연산가능한 여러개의 다항식으로 쪼개놓는 trick

  ```python
  exp(   log(   exp(800) + exp(900)   )   ) # overflow
  => log 내부 연산 exp(800) + exp(900) 이 너무 크므로 overflow
  => log(exp(800) + exp(900)) = log(exp(800) ( exp(0) + exp(100) ) )
  = 800 + log( exp(0) + exp(100) )
  => log 내부 연산이 작아졌으므로 900.0 return !
  ```



* auto encoder처럼 reconstruction에 대한 error를 측정할 때 사용된다.



`pred = [p1, p2, p3, ...]` , `true = [t1, t2, t3 ... ]` 일 때

`L(pred, true) = reduction( -wi(  ti * log(pi) + (1-ti) * log(1-pi)  ) )`

* reduction은 default가 mean이고 sum을 줄 수도 있다.
* wi는 `weigth = [w1, w2, w3, ... ]` 
  * multilabel classification일 때 
