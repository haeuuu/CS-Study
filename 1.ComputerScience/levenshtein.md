# Levenshtein 편집 거리 구하기

>  DP를 이용해서 편집에 걸리는 최소 횟수를 계산한다.



두 문자열이 주어지고, 삽입/삭제/교체만이 가능할 때의 최소 편집 횟수를 구하고자 한다.

2차원 배열을 이용해서 + DP로 채워나가면 된다.



## step0. DP를 위한 table 정의하기.

`dist['string1']['sting2']` = `string1`이 `string2`가 되기 위해 필요한 최소 편집 거리가 되도록 `dist`행렬을 만들 것이다.

다음은 `gzuab`를 `abc`로 바꾸는 데에 필요한 최소 편집 횟수를 구하기 위해 정의된 행렬`dist`이다.

|           | NULL                                                  | a    | ab                               | abc  |
| --------- | ----------------------------------------------------- | ---- | -------------------------------- | ---- |
| **NULL**  |                                                       |      |                                  |      |
| **g**     |                                                       |      |                                  |      |
| **gz**    | gz가 빈 문자열이 되기 위해<br />필요한 최소 편집 횟수 |      |                                  |      |
| **gzu**   |                                                       |      |                                  |      |
| **gzua**  |                                                       |      | gzua가 ab가 되기 위해 필요한 ... |      |
| **gzuab** |                                                       |      |                                  |      |



## step1. 먼저 `NULL` 즉 빈 문자열에서 시작하는 경우부터 고려해보자.

빈 문자열에서 길이 n인 어떤 문자열이 되기 위해서는 n번의 삽입을 거치는 것이 최소 편집 횟수가 된다.

`dist[0][i]` 와 `dist[i][0]` 을 모두 0부터 length가 되도록 채워준다.

|           | NULL | a    | ab   | abc  |
| --------- | ---- | ---- | ---- | ---- |
| **NULL**  | 0    | 1    | 2    | 3    |
| **g**     | 1    |      |      |      |
| **gz**    | 2    |      |      |      |
| **gzu**   | 3    |      |      |      |
| **gzua**  | 4    |      |      |      |
| **gzuab** | 5    |      |      |      |



## step2. abe를 c로 만드는 연산은 <u>abe를 NULL로 만드는 연산 + alpha</u>이다.

이 연산이 DP가 될 수 있는 이유가 중요하다. 이전의 연산 값을 어떻게 이용할 수 있을지 경우를 나누어보자.

가장 마지막에 어떤 연산을 수행할 것이냐에 따라 다음 4가지 케이스로 나눌 수 있다.



##### case 1 : 삽입

abe를 c로 만드는 횟수를 계산하기 위해 처음부터 해 볼 필요는 없다.

abe를 c로 만들기 위해서는 1. abe를 NULL로 만들고 2. c를 삽입한다. 의 절차를 거쳐야 한다.

abe를 NULL로 만든다? 이미 `dist[abe][NULL]` 즉 나의 왼쪽 칸에서 이미 계산된 값에 1만 더해주면 된다. (insert 수행을 위해)

즉 abe를 NULL로 만드는 횟수 + c를 삽입하는 횟수 =  `dist[i-1][j]+1`



##### case 2 : 삭제

abef를 ogh로 만드는 연산은 어떨까? abef를 oghf로 만들고 나서, 마지막 글자인 f만 삭제해주면 된다.

즉 abe를 ogh로 만드는 연산 횟수 + f를 삭제하는 연산 횟수 = `dist[i][j-1] + 1`



##### case 3 : 교체

abef를 oghj로 만드는 연산은 abef를 oghf로 만든 후에 f를 j로 교체하면 된다.

abe를 ogh로 만드는 연산 횟수 + f를 j로 교체하는 횟수 1 = `dist[i-1][j-1]+1`



##### case 4 : 동시에 추가되는 경우

abef를 oghf로 만드는 연산은 어떨까? f는 이미 둘다 가지고 있으므로 고려하지 않아도 된다.

abe를 ogh로 만드는 데에 필요한 횟수만 안다면, 이후에는 양쪽의 문자열에 모두 f만 붙여주면 된다.

즉 `dist[abef][oghf] == dist[abe][ogh]` 



## step3. 점화식 세우기

임의의 두 문자열에 대해서도 가능성 1 ~ 4 중에 하나만을 고려하면 될까?

가능성 4는 따로 보아야 한다. 마지막 글자가 같을 때만 이전 결과의 횟수를 그대로 가져올 수 있기 때문이다. 그리고 당연히 이 경우에는 가능성 1 ~ 4중에서 4가 가장 최소 연산 횟수이다.

마지막 글자가 다르다면 가능성 1 ~ 3을 모두 고려해야 한다. abef를 ogh로 만들기 위한 방법에는 다음 세가지가 있다.

1. abef를 og로 만들고 h를 삽입한다.
2. abef를 oghf로 만들고 f를 삭제한다.
3. abef를 ogf로 만들고 f를 h로 교체한다.

어떤 것이 최소 횟수일까? 직접 해보기 전까지는 알 수 없으므로 min을 이용하자.

```python
if 마지막 글자가 같다면:
    dist[i][j] = dist[i-1][j-1]
else:
    # 마지막 글자를 교체하거나, 삭제하거나, 삽입한다.
    dist[i][j] = min(dist[i-1][j-1]+1, dist[i][j-1]+1, dist[i-1][j]+1)
```



## python으로 구현하기

```python
def edit(source, target): # source를 target으로 바꾸기 위해 필요한 최소 연산 횟수를 계산한다.

    dist = [[0] * (len(edit_target)+1) for _ in range(len(edit_source)+1)]
    
    # 0번째 열/행을 0~length로 채워주기
    for i in range(len(edit_target)+1):
        dist[0][i] = i
    for i in range(len(edit_source)+1):
        dist[i][0] = i

    for i in range(1,len(edit_source)+1):
        for j in range(1,len(edit_target)+1):
            if edit_source[i-1] == edit_target[j-1]:
                # 마지막 글자가 같다면 이전 연산값을 그대로 가져온다.
                dist[i][j] = dist[i - 1][j - 1]
            else:
                # 변환하거나, 마지막 글자를 추가하거나, 마지막 글자를 삭제하거나
                dist[i][j] = min(dist[i - 1][j - 1] + 1,dist[i][j - 1] + 1, dist[i - 1][j] + 1)

    # 구해진 행렬 확인하기
    for ls in dist:
        print(*ls)

    return dist[-1][-1]
```

```python
print(edit('gzuab','abc')) # gzu를 abc로 만드는 데에 걸리는 최소 연산 횟수는 ?
```

