# Word2Vec

같이 자주 등장하는 단어는 서로 연관이 있지 않을까? 그 의미까지 유사하지 않을까?

1. **CBOW(Continuous Bag of Words)** 단어 하나를 가리고, 주변 정보를 통해 추측하도록 하자.

   `나는 [   ]에 간다.` => `나는 [학교]에 간다.`

2. **Skip-Gram** 중요 단어 몇개만 보고, 주변 정보를 추측하도록 하자.

   `[ A ] 학교 [ B ]` => `A ; 선생님, 친구, ...` `B ; 간다. 공부한다. ...`



### CBOW v.s. Skip-Gram

The quick brown fox jumps over the lazy dog. 에 대해 학습한다고 생각해보자.

1 iter당 brown이 update되는 횟수는 CBOW는 한 번 뿐이지만 Skip-Gram은 여러번 가능하다.

=> brown이 중심단어가 될 수 있는 기회가 몇 번인가? 를 생각해보면 쉽다 !



* **CBOW** brown을 가리고 학습한 후 brown을 update한다. 딱 한 번의 업데이트 기회만 갖는다.

* **Skip-Gram**  window size가 2일 때 brown이 중심단어가 되는 경우는 총 4번 : (the, brown), (quick, brown), (brown, fox) (brown, jumps)

  즉 update 역시 4번 일어난다.

  > window size가 약간 헷갈림 ㅠ

말뭉치 크기가 동일 할 때, window size가 2인 경우만 해도 학습량이 네 배 차이가 난다 !



2번 방법에 대해 기술한다.



## Architecture

vocab의 크기 V, hidden layer의 dimension H에 대해

1. 중심단어인 input vector (one hot vector) 1xV가 들어오면
2. W~VxH~ 에 의해 1xH 차원으로 embedding되고
3. 다시 W~HxV~ 에 의해 1xV차원으로 embedding된다.



## objective funtion

중요 단어 C가 주어졌을 때, 중심단어 O가 나타날 확률을 최대화하자. 즉 중심단어만 봐도 주변 단어를 잘 맞추도록 만들자!

이 때 조건부 확률 식은 두 벡터의 내적 + exponential을 사용한다.

1. 내적을 통해 두 벡터의 코사인 유사도를 반영한다,
2. exponential을 통해 유사도가 큰 두 벡터의 점수를 지수적으로 증가시켜준다.
   (내적이 작은 단어는 더 작도록, 큰 단어는 더 크도록 만들어준다.)

![image-20201127163518124](../fig/image-20201127163518124.png)





## Trick

> 등장하는 확률값은 고정값이므로 학습 전에 미리 계산해놓는다.



### 1 ) Subsampling : 자주 나오는 단어는 가끔 <u>제외</u>하고 학습하자! 학습량 줄이기

10만개의 단어 중에서 조사(은,는,이,가)는 아마도 책상보다는 훨~씬 많이 등장할 것이다. 즉 학습될 기회가 많다.

=> 정보도, 기회도 많은 단어들은 학습을 좀 덜해도 괜찮지 않을까?

=> 자주 등장하는 단어의 학습량을 확률적으로 줄이자 !



i번째 단어가 학습에서 **제외될 확률**을 다음과 같이 정의한다.

f(w~i~)는 단어의 등장 빈도/전체 등장 단어수 이다. t는 hyperparameter지만 0.00001을 권장한다.

![image-20201127175121291](../fig/image-20201127175121291.png)

### 2 ) Negative sampling : 꼭 매번 10만개를 다 봐야해? 계산량 줄이기

역시 vocab size에 의한 문제를 해결하고자 하는 방법이다.

objective function의 분모값을 계산하기 위해서 10만번의 내적과 exponential을 1 iter 내에서도 무수히 많이 해야한다!

=> 실제 정답 단어는 겨우 c개이며 negative sample(window size내에 등장하지 않은 단어)는 10만개 미만이다.

=> negative sample중에서 몇 개만 뽑아쓰는건 어때?



window size 내에 등장하지 않는 단어를 5~20개 정도 뽑는다. (negative sample)

정답 단어 c개 (positive sample)과 합쳐서 딱 요만큼만을 이용해 확률값을 계산한다!



i번째 단어가 negative sample로 뽑힐 확률은

![image-20201127175841951](../fig/image-20201127175841951.png)



###### 랜덤하게 뽑지 않고 빈도를 고려한 이유는 무엇일까? 3/4승에는 어떤 효과가 있을까?

> random v.s. freq에 대한 개인적인 생각
>
> 자주 등장해서 학습의 기회가 많았던 단어들은 아마 학습이 잘 되었을 것이다. embedding도 비교적 엉뚱하지 않을 것이다.
>
> 믿을만한 embedding을 이용해서 현재 단어를 update하는 것이 더 적절하지 않을까?



* 그냥 랜덤 : 모두 뽑힐 확률이 같다.
* 그냥 빈도 : 많이 등장할 수록 뽑힐 확률이 크다.
* 빈도^(3/4)^ : 적게 등장한 단어라도 지수승을 통해 뽑힐 확률을 약간 높여준다.
  * f(wi) = 0.9인 경우 f(wi)^(3/4)^ = 0.92 => 크게 달라지지 않음
  * f(wi) = 0.09인 경우 f(wi)^(3/4)^ = 0.16 => 상대적으로 큰 폭 상승함.



## 추가 메모

* hidden layer는 1개 뿐이다.  구조가 간단하다.
* 비선형 활성화 함수를 쓰지 않았다. 사실상 선형 모델이다.



## Reference

[ratsgo's blog](https://ratsgo.github.io/from%20frequency%20to%20semantics/2017/03/11/embedding/)

[ratsgo's blog](https://ratsgo.github.io/from%20frequency%20to%20semantics/2017/03/30/word2vec/)

[NNLM to W2V](https://shuuki4.wordpress.com/2016/01/27/word2vec-%EA%B4%80%EB%A0%A8-%EC%9D%B4%EB%A1%A0-%EC%A0%95%EB%A6%AC/)
