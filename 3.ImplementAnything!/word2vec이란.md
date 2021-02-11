# Word2Vec

같이 자주 등장하는 단어는 서로 연관이 있지 않을까? 그 의미까지 유사하지 않을까?

1. **CBOW(Continuous Bag of Words)** 단어 하나를 가리고, 주변 정보를 통해 추측하도록 하자.

   `나는 [   ]에 간다.` => `나는 [학교]에 간다.`

2. **Skip-Gram** 중요 단어 몇개만 보고, 주변 정보를 추측하도록 하자.

   `[ A ] 학교 [ B ]` => `A ; 선생님, 친구, ...` `B ; 간다. 공부한다. ...`



중심 단어 c와 주변 단어 w가 '같이 등장했다.'고 말할 수 있으려면, w는 window size내에 있어야 한다. 즉

`The quick brown fox jumps over the lazy dog.` 에서 중심단어가 `fox`일 때, `over`는 window size가 1일때는 주변 단어가 아니지만 2가 되면 주변 단어가 된다.



### CBOW v.s. Skip-Gram

The quick brown fox jumps over the lazy dog. 에 대해 학습한다고 생각해보자.

1 iter당 brown이 update되는 횟수는 CBOW는 한 번 뿐이지만 Skip-Gram은 여러번 가능하다.

=> brown이 중심단어가 될 수 있는 기회가 몇 번인가? 를 생각해보면 쉽다 !



* **CBOW** brown을 가리고 학습한 후 brown을 update한다. 딱 한 번의 업데이트 기회만 갖는다.

* **Skip-Gram**  window size가 2일 때 brown이 중심단어가 되는 경우는 총 4번 : (the, brown), (quick, brown), (brown, fox) (brown, jumps)

  즉 update 역시 4번 일어난다.

  > window size가 약간 헷갈림 ㅠ

말뭉치 크기가 동일 할 때, window size가 2인 경우만 해도 학습량이 네 배 차이가 난다 !



다음은 쭉 skip gram을 가정하고 기술한다. CBOW는 input이 여러개, output이 한개인 버전으로 약간 수정해주면 된다.



## Architecture

> ##### :raising_hand: 중심 단어 하나만 알려줄테니까, 주변 단어까지 채워서 return해줘!



![image-20201128161454000](../fig/image-20201128161454000.png)



word2vec의 목적은 단어의 embedding이 되는 W, W*를 잘 학습하는 것이다.

embedding이 잘 학습되었다는 것은 어떻게 정의할 수 있을까?

비슷한 단어의 cosine 유사도는 크고, 상관이 없는 단어일 수록 그 값이 작으면 좋겠다 !

즉 더위와 여름은 유사한 vector를, 더위와 핸드폰은 다른 방향의 vector를 가져야 한다 !



모델 구조는 굉장히 간단한데, 어떠한 비선형 activation도 없이, 그저 두 번의 행렬 곱(W, W*)만 거친다.

심지어 첫번째 연산은 사실상 내적도 아니다. input이 one hot vector이기 때문에 `W*input = w[input_index]` 즉 고작 indexing일 뿐이다.

그러므로 모델의 output은 그저 W의 행벡터 하나와 W* 의 내적일뿐이다.



기본적인 흐름은 다음과 같다.

1. `fox`에 해당하는 embedding `v_c` 를 추출한다.
2. 내 vocab에 있는 모든 단어 `u_i` 와 내적하여 유사도를 계산한다.
   * `fox`가 중심단어일 때 `w_i`가 등장할 score 정도로 생각할 수 있다.
3. 내적한 값을 softamax에 통과시켜 확률값을 얻는다.
   * `fox`의 주변에 있던 단어인 `jumps`, `brown`이 높은 확률값을 가지도록 학습되어야 한다.
4. loss를 계산하고 backprop한다.

학습이 끝난 후 어떤 행렬을 사용해도 되지만 일반적으로는 앞에 있는 W 행렬을 최종 embedding으로 사용한다.



### objective funtion

softmax를 지나고 나온 값을 "중심단어 c가 주어졌을 때, 주변단어 w가 나올 확률"로 생각한다면,
$$
P(w|c) = \exp(v_w^Tv_c) / \Sigma\exp(v_i^Tv_c)
$$
cross entropy 식에 의해서 자연스럽게 조건부 확률들을 최대화 하도록 학습됨을 알 수 있다.
$$
argmin -\Sigma log(P(v_i^Tv_c))
$$


## 연산량을 줄이기 위한 Trick

> 등장하는 확률값은 고정값이므로 학습 전에 미리 계산해놓는다.

word2vec의 연산량은 vocab size에 영향을 받는다. 어떻게 하면 줄일 수 있을까?



### 1 ) Subsampling : 자주 나오는 단어는 가끔 <u>제외</u>하고 학습하자! 학습량 줄이기

10만개의 단어 중에서 조사(은,는,이,가)는 아마도 책상보다는 훨~씬 많이 등장할 것이다. 즉 학습될 기회가 많다.

=> 정보도, 기회도 많은 단어들은 학습을 좀 덜해도 괜찮지 않을까?

=> 자주 등장하는 단어의 학습량을 확률적으로 줄이자 !



i번째 단어가 학습에서 **제외될 확률**을 다음과 같이 정의한다.

f(w~i~)는 단어의 등장 빈도/전체 등장 단어수 이다.



t == f(wi)가 되면 p(wi) = 0이므로 i번째 단어는 무조건 학습한다.

f(wi)가 t보다 커지면 커질 수록 제외될 확률이 높아질 것이고, 반대의 경우 낮아진다.

즉 t는 threshold 역할을 함을 알 수 있다. 논문에서는 0.000001을 사용했을 때 가장 좋은 결과를 얻었다고 한다.

![image-20201127175121291](../fig/image-20201127175121291.png)



### 2 ) Negative sampling : 꼭 매번 10만개를 다 봐야해? 계산량 줄이기

objective function의 분모값을 계산하기 위해서 10만번의 내적과 exponential을 1 iter 내에서도 무수히 많이 해야한다!

=> 실제 정답 단어는 겨우 c개이며 negative sample(window size내에 등장하지 않은 단어)는 10만개 미만이다.

=> negative sample중에서 몇 개만 뽑아쓰는건 어때?



window size 내에 등장하지 않는 단어를 5~20개 정도 뽑는다. (negative sample)

정답 단어 c개 (positive sample)과 합쳐서 딱 요만큼만을 이용해 확률값을 계산한다!



i번째 단어가 negative sample로 뽑힐 확률은

![image-20201127175841951](../fig/image-20201127175841951.png)



#### Negative sampling을 적용했을 때 objective funtion

> 기본적인 skip gram 모델은 p(w|c)를 모델링하지만 Negative sampling을 적용하면 약간 달라진다.
>
> [더 자세한 설명이 필요하다면 paper를 읽어보자](https://arxiv.org/pdf/1402.3722.pdf)

`D` ; positive set, `D_tilde` ; negative set

word w, context c, W = u, W* = v에 대해

1. `P(D=1|w,c)` ; `(w,c)`가 positive set에 있을 확률 = `sigmoid(u_v.T*v_c)` 로 정의하자.

<img src="../fig/image-20201128044751851.png" alt="image-20201128044751851" style="zoom: 67%;" />

2. positive set의 확률은 크게, negative set의 확률은 작게 만들자.

<img src="../fig/image-20201128042830785.png" alt="image-20201128042830785" style="zoom:67%;" />

3. 위 식을 다시 sigmoid를 씌워 정리하면

<img src="../fig/image-20201128180258576.png" alt="image-20201128180258576" style="zoom:67%;" />



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



##### 

###### Negative sampling 외에, softmax의 연산량을 줄이기 위한 방법들

##### [Hierarchical Softmax](https://yjjo.tistory.com/14?category=876638)

Huffman Tree(Binary)를 이용해서 path를 타고 가며 연산량을 줄인다.

논문 후반부의 실험 결과에 따르면 Negative sampling의 성능이 더 좋았다 (Analogy Reasoning Task. 한국-서울+일본 = 도쿄). 

물론 HS가 더 좋은 성능을 내는 task도 있다.



##### [NCE-K](https://yjjo.tistory.com/14?category=876638)

Noise Contrastive Estimation

Negative sampling과 유사하다. (NCE 내에 Negative sampling이 있는건지 그냥 이 개념을 가지고 와서 새롭게 만든건지 찾아봐야겠다)





## 추가 메모

* 비선형 활성화 함수를 쓰지 않았다. 사실상 선형 모델이다.



## Reference

[ratsgo's blog](https://ratsgo.github.io/from%20frequency%20to%20semantics/2017/03/11/embedding/)

[ratsgo's blog](https://ratsgo.github.io/from%20frequency%20to%20semantics/2017/03/30/word2vec/)

[NNLM to W2V](https://shuuki4.wordpress.com/2016/01/27/word2vec-%EA%B4%80%EB%A0%A8-%EC%9D%B4%EB%A1%A0-%EC%A0%95%EB%A6%AC/)
