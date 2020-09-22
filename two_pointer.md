# :eight_pointed_black_star: two pointer

> [설명 굿굿](https://blog.naver.com/kks227/220795165570)



길이 N짜리인 배열에서, 부분 배열의 합이 M이 되는 경우는 몇가지일까?

1 ) 모든 경우의 수를 체크한다.

* 길이 1짜리인 배열 `n`개
* 길이 2짜리인 배열 `n-1`개 
* ...
* 길이 n짜리인 배열 `1`개

즉 `n + n-1 + ... + 1` 가지 경우의 수에 대해 확인해야하는데, 길이 N짜리인 배열의 합을 구하는 시간복잡도는 O(N)이다. 이를 O(1)이라고 치더라도 최소 O(N^2^)이 소요된다.



만약 **배열의 모든 원소와 M이 모두 양수라면** 두개의 pointer를 통해 해결할 수 있다.

1. `start = end = 0`으로 시작한다. 단 항상 `start <= end`이다.

   `[start, end]`로 할건지 `[start,end)`로 할건지는 본인 마음이지만 나는 전자를 전제로 구현했다.

2. 만약 **현재 부분합이 M보다 크거나 같다** or **`end == N-1`** 라면

   > end가 N에 도달한 경우에는 start만 늘려주면서 **start == N-1 가 되는 순간까지 M이 되는 경우가 있는지 판단해줘야함 !!!!!**

   1. 다른 경우의 수를 탐색하기 위해 `start += 1`
   2. 이때 M과 같은 경우에만 `answer += 1`을 해준다.

   그게 아니면 더 더해주어야하므로 `end += 1`

3. `end == N`이 된 이후에 start만 계속 늘려주다가 start가 N-1에 도달하면 중지한다.



위 알고리즘은 end도 N에, start도 N에 갈때까지 각각 O(N)이 필요하므로 **총 시간 복잡도도 O(N) !**



## 구현

##### 몇가지 주의사항

1. :star: **종료 조건은 start에 달려있다.**
2. 이상 이하, 즉 `[start, end]` 로 구현되어있다.
   * 초기 `curr_sum`은 `0`이 아니라 `arr[0]`다 !
   * if문에서 `end == N (x) N-1 (o)`
3. start를 늘릴 때는 `curr_sum -= arr[start]`를 먼저 하고, end를 늘릴 때는 `curr_sum += arr[end]`를 나중에 해야겠지 당연히 !

```python
def solution(arr, M):
    N = len(arr)
    start, end, curr_sum = 0, 0, arr[0]
    answer = 0
    
	################ start가 끝에 도달할 때까지 계속 진행한다. (start <= end 가 아님에 주의하자.)
    while start < N:
    #################

    	# 1. start를 늘리는 조건 : end가 끝에 도달했거나, 현재 sum이 M이상일 때
    	if curr_sum >= M or end == N-1:
            if curr_sum == M:
                answer += 1
            curr_sum -= arr[start]
            start += 1
        
        # 2. end를 늘린다. (위 조건을 end == N 으로 하면 end += 1후 indexing 과정에서 error ! )
        else:
            end += 1
            curr_sum += arr[end]

    return answer
```



##### `while start <= end가 아닌 이유는 !`

만약 길이 1짜리에서 합이 M이 된다면, 다음 turn에서는 `start > end` 가 된다.

(이상 이하니까 `start == end`인 상태에서 `start +=1`만 되니까 )

`start>end`인 경우에 view는 빈리스트 즉 `[]`이고, 다시 길이 1짜리부터 탐색을 이어나가야한다 !!!

종료 조건은 start에 달려있다는 것을 잊지 말자아 ~



## 2003 수들의 합 2

```python
4 5
view :[2, 5], curr_sum :7
-- 5 이상이므로 start를 늘립니다.

5 5
view :[5], curr_sum :5
-- 5 이상이므로 start를 늘립니다.
---- M과 같으므로 answer += 1

6 5 # 만약 while start <= end라면 arr를 탐색하다가 중간에 중지하게 됨 !
view :[], curr_sum :0
-- 5 이하이므로 end를 늘립니다.

6 6
view :[3], curr_sum :3
-- 5 이하이므로 end를 늘립니다.
```

```python
# 위 결과를 얻으려면

def solution(arr, M):
    N = len(arr)
    start, end, curr_sum = 0, 0, arr[0]
    answer = 0

    while start < N:
        print(start,end)
        print(f'view :{arr[start:end+1]}, curr_sum :{curr_sum}')
        if curr_sum >= M or end == N-1:
            print(f"-- {M} 이상이므로 start를 늘립니다.")
            if curr_sum == M:
                print('---- M과 같으므로 answer += 1')
                answer += 1
            curr_sum -= arr[start]
            start += 1
        else:
            print(f"-- {M} 이하이므로 end를 늘립니다.")
            end += 1
            curr_sum += arr[end]
        print()

    return answer

print(solution([1, 2, 3, 4, 2, 5, 3, 1, 1, 2],5))
```



# :sunrise: [비슷한 테크닉] sliding window

two pointer 알고리즘에서 언제나 `end-start`가 일정한 조건이 추가된다면 ! 슬라이딩 윈도우 !



## 2075 N번째 큰 수

> N×N의 표에 수 N^2^개 채워져 있다. 채워진 수에는 한 가지 특징이 있는데, 모든 수는 자신의 한 칸 위에 있는 수보다 크다는 것이다. N=5일 때의 예를 보자.
>
> | 12   | 7    | 9    | 15   | 5    |
> | ---- | ---- | ---- | ---- | ---- |
> | 13   | 8    | 11   | 19   | 6    |
> | 21   | 10   | 26   | 31   | 16   |
> | 48   | 14   | 28   | 35   | 25   |
> | 52   | 20   | 32   | 41   | 49   |
>
> 이러한 표가 주어졌을 때, N번째 큰 수를 찾는 프로그램을 작성하시오. 표에 채워진 수는 모두 다르다. N(1 ≤ N ≤ **1,500**)



### 크기가 N으로 고정된 min heap을 이용하자

> N개가 들어올 때마다 정렬한 후 다시 크기가 N이 되도록 slicing하는 알고리즘도 통과하는걸 보니
>
> 이 문제의 의도는 **N번째 수를 알기 위해 N^2^ 개를 모두 저장하지 말아라 !** 인듯 하다.



메모리 제한이 있기 때문에 ,,, **heap의 크기를 N으로 제한**한다. (나름 window ,,라고 볼 수 있겠다.)

1. len(heap) < N이면 `push`

2. `==N` 이면 일단 지금의 `value`를 push한 다음, smallest를 `pop`

   * 이렇게 끝까지 반복하면

   * `[n번째 큰 수, n-1번째 큰 수, ... , 1번째 큰 수]` 만 남게된다 !!

     ```python
     1*n개까지의 [n등 ~ 1등] heap : [5, 7, 9, 15, 12]
     2*n개까지의 [n등 ~ 1등] heap : [11, 12, 19, 15, 13]
     3*n개까지의 [n등 ~ 1등] heap : [16, 21, 19, 26, 31]
     4*n개까지의 [n등 ~ 1등] heap : [26, 28, 48, 35, 31]
     5*n개까지의 [n등 ~ 1등] heap : [35, 41, 48, 49, 52] # => 35가 최종 답 !
     ```

     

```python
from heapq import *
import sys

f = lambda:sys.stdin.readline().split()
N = int(input())

arr  = []

for _ in range(N):
    for value in f():
        
        if len(arr) < N:
            heappush(arr, int(value))
        else:
            heappushpop(arr,int(value))

print(arr[0])
```



## 2096 내려가기

N*3 지도 위에 0~9 숫자가 적혀있다. 현재 칸에서는 1. 바로 아래칸 2. 바로 아래칸과 인접한 칸 으로만 이동할 수 있다.

이 경우 얻을 수 있는 최대/최소 점수는?



#### 만약 메모리 제한이 없었다면 DP!

```python
N = int(input())
arr = [[*map(int,input().split())] for _ in range(N)]

max_score = [[0,0,0] for _ in range(N)]
max_score[0] = [i for i in arr[0]]

min_score = [[0,0,0] for _ in range(N)]
min_score[0] = [i for i in arr[0]]

for floor in range(1,N):
    for j in range(3):
        
        # max에 대한 DP
        max_score[floor][j] = arr[floor][j] + \
                              max(max_score[floor-1][j + dj] for dj in [-1,0,1] if 0<=j+dj<3)

        # min에 대한 DP
        min_score[floor][j] = arr[floor][j] + \
                              min(min_score[floor-1][j + dj] for dj in [-1,0,1] if 0<=j+dj<3)

print(max(max_score[-1]), min(min_score[-1]))
```



#### 사실 직전 배열 정보만 알면 돼 !

하지만 계속 O(3)으로 배열을 복사해야한다.

```python
N = int(input())
arr = [[*map(int,input().split())] for _ in range(N)]

max_score = [i for i in arr[0]]
min_score = [i for i in arr[0]]

for floor in range(1,N):
    temp_max_score = [0,0,0]
    temp_min_score = [0,0,0]

    for j in range(3):
        temp_max_score[j] = arr[floor][j] + \
                              max(max_score[j + dj] for dj in [-1,0,1] if 0<=j+dj<3)

        temp_min_score[j] = arr[floor][j] + \
                              min(min_score[j + dj] for dj in [-1,0,1] if 0<=j+dj<3)

    max_score = [i for i in temp_max_score]
    min_score = [i for i in temp_min_score]


print(max(max_score), min(min_score))
```



이렇게 하면 훨~씬 깔끔 ! + 시간이 10분의 1로 단축 !

```python
N = int(input())
arr = [[*map(int,input().split())] for _ in range(N)]

M1,M2,M3 = 0,0,0
m1,m2,m3 = 0,0,0

for i in range(N):
    M1,M2,M3 = max(M1,M2) + arr[i][0], max(M1,M2,M3) + arr[i][1], max(M2,M3) + arr[i][2]
    m1,m2,m3 = min(m1,m2) + arr[i][0], min(m1,m2,m3) + arr[i][1], min(m2,m3) + arr[i][2]

print(max(M1,M2,M3), min(m1,m2,m3))
```



## 2230 수 고르기

### 1 ) 완전탐색 O(N^2^)

당여어으어어언히 `시간초과 !`

```python
from itertools import combinations

N,M = map(int,input().split())
arr = [int(input()) for _ in range(N)]
answer = float('inf')

for i,j in combinations(range(N),2):
    if abs(arr[i] - arr[j]) >= M:
        answer = min(answer, abs(arr[i] - arr[j]))

print(answer)
```



### 2 ) sort후 two pointer O(NlogN)

* stdin으로 받아야 시간 줄어든다 ~



#### :thinking: 내가 놓친 부분 => <u>EndState를 end 기준으로 본다면</u> loop가 훨씬 일찍 끝난다.

> **일찍 끝낼 수 있는 이유는 <u>정렬했기 때문</u>** !!!!! 가보지 않고는 정보를 알 수 없다면 꼭 다 돌아야한다.

나는 `start < N`까지 돌렸는데 다들 `end < N` 까지만 돌렸다. 왜일까 ?

* `end += 1`은 현재의 `end-start`가 M보다 작거나 같을 때 일어난다.

* 만약 계속 end가 늘어나서, 다음처럼 end가 먼저 벽에 닿았다고 생각해보자.

  * start 이후의 값은 모두 `arr[start]` 보다 작다 !

  * **그러므로 start를 더 늘려서 검사해봤자, <u>answer의 후보가 될만한 값이 없다</u>.**

    | index | 0    | 1    | 2    | 3    | 4 :star: | 5    | 6    | ...  | 100 :moon: |
    | ----- | ---- | ---- | ---- | ---- | -------- | ---- | ---- | ---- | ---------- |
    | value | 10   | 15   | 20   | 21   | **45**   | 67   | 89   | ...  | **999**    |

    정렬했기 때문에 45 이후의 값들은 당연히 `diff = 999 - 45` 보다 작은 diff를 갖는다.



##### 수정된 풀이

```python
N,M = map(int,input().split())
arr = [int(input()) for _ in range(N)]
arr.sort()

start,end = 0,1
answer = float('inf')

while start < N and end < N:
    curr_diff = arr[end] - arr[start]

    if curr_diff >= M:
        start += 1
        answer = min(answer,curr_diff)
    else:
        end += 1

print(answer)
```



##### 이전 풀이

```python
N,M = map(int,input().split())
arr = [int(input()) for _ in range(N)]
arr.sort()

start,end = 0,1
answer = float('inf')

while start < N:
    curr_diff = arr[end] - arr[start]

    if curr_diff >= M or end == N-1:
        start += 1
        if curr_diff >= M:
            answer = min(answer,curr_diff)
    else:
        end += 1

print(answer)
```



### 3 ) binary search

#### :red_circle: 예외 처리 잘하기 !! 안하면 RE ㅠㅠ

만약 `arr = [1,2,3,10]`에

1.  `bisect_left`로 5를 넣고자 한다면 `3`을 return한다.
2.  `20`을 넣고자 한다면? `length i.e. 4`를 return한다 !!! **여기를 처리 안하면 RE** !

```python
from bisect import bisect_left
import sys
f = lambda: sys.stdin.readline()

N,M = map(int,f().split())
arr = [int(f()) for _ in range(N)]
arr.sort()
answer = float('inf')

for start in arr:
    end = bisect_left(arr, start + M)
    
    ############ end == N을 return했다는 것은 start와 M이상 차이나는 원소가 없다는 뜻이 되므로 곧바로 중지한다.
    if end >= N:
        break
    ############
    
    answer = min(answer, arr[end] - start)
    
print(answer)
```

