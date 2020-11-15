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
                dist[i][j] = min(dist[i - 1][j - 1] + 2,dist[i][j - 1] + 1, dist[i - 1][j] + 1)

    for ls in dist:
        print(*ls)

    return dist[-1][-1]

print(edit('gzuab','abc')) # gzu를 abc로 만드는 데에 걸리는 시간은!
```



`dist['string1']['sting2']` = `string1`이 `string2`가 되기 위해 필요한 최소 편집 거리가 되도록 `dist`행렬을 만들 것이다.

|           | NULL                                             | a    | ab                               | abc  |
| --------- | ------------------------------------------------ | ---- | -------------------------------- | ---- |
| **NULL**  |                                                  |      |                                  |      |
| **g**     |                                                  |      |                                  |      |
| **gz**    | gz가 빈 문자열이 되기 위해 필요한 최소 편집 횟수 |      |                                  |      |
| **gzu**   |                                                  |      |                                  |      |
| **gzua**  |                                                  |      | gzua가 ab가 되기 위해 필요한 ... |      |
| **gzuab** |                                                  |      |                                  |      |



##### 먼저 `NULL` 즉 빈 문자열에서 시작하는 경우부터 고려해보자.





|           | NULL | a    | ab   | abc  |
| --------- | ---- | ---- | ---- | ---- |
| **NULL**  |      |      |      |      |
| **g**     |      |      |      |      |
| **gz**    |      |      |      |      |
| **gzu**   |      |      |      |      |
| **gzua**  |      |      |      |      |
| **gzuab** |      |      |      |      |

