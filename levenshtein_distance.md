# 편집 거리 알고리즘

DP를 이용해서 편집에 걸리는 최소 횟수를 계산한다.

```python
def edit(source, target):

    dist = [[0] * (len(edit_target)+1) for _ in range(len(edit_source)+1)]
    
    for i in range(len(edit_target)+1):
        dist[0][i] = i
    for i in range(len(edit_source)+1):
        dist[i][0] = i

    for i in range(1,len(edit_source)+1):
        for j in range(1,len(edit_target)+1):
            if edit_source[i-1] == edit_target[j-1]:
                dist[i][j] = dist[i - 1][j - 1]
            else:
                # 변환하거나, 마지막 글자를 추가하거나, 마지막 글자를 삭제하거나
                dist[i][j] = min(dist[i - 1][j - 1] + 1,dist[i][j - 1] + 1, dist[i - 1][j] + 1)

    for ls in dist:
        print(*ls)

    return dist[-1][-1]

print(edit('gzuab','abc')) # gzu를 abc로 만드는 데에 걸리는 시간은!
```



`dist['string1']['sting2']` = `string1`이 `string2`가 되기 위해 필요한 최소 편집 거리가 되도록 `dist`행렬을 만들 것이다.

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

이 연산이 DP가 될 수 있는 이유가 중요하다.

이미 이전에 계산한 값을 이용해서 구할 수 있고 이는 다음 세가지 가능성 중에서 가장 작은 값으로 결정된다.



##### 가능성 1 : 삽입

abe를 c로 만드는 횟수를 계산하기 위해 처음부터 해 볼 필요는 없다.

abe를 c로 만들기 위해서는 1. abe를 NULL로 만들고 2. c를 삽입한다. 의 절차를 거쳐야 한다.

abe를 NULL로 만든다? 이미 `dist[abe][NULL]` 즉 나의 왼쪽 칸에서 이미 계산된 값에 insert 수행을 위해 1만 더 더해주면 되는 것이다 !

abcde에서 abefg를 만드는 연산 역시 abcde에서 abef를 만든 후에 g만 추가해주면 된다.



##### 가능성 2 : 삭제

abef를 abe로 만드는 연산은 어떨까? abe를 abe로 만들고 나서, 마지막 글자인 f만 삭제해주면 된다.



##### 가능성 3 : 교체

abef를 abeg로 만드는 연산은 abe



