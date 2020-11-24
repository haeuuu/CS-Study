bot에게 rational action을 가르쳐주는 문제들

좌표 등을 다루고, 구현력을 키울 수 있다!



> 문제 요약
>
> 헷갈렸던 점, 디버깅이 오래걸린 이유
>
> 다른 사람의 풀이
>
> 배워갈 점



# Bot saves princess

> Bot `m` 이 princess `p` 로 가기 위한 최단 경로를 출력하라.



##### 헷갈렸던 점

1. `(x,y)`로 생각하고 일반 좌표계에 넣음. 하지만  `arr[i][j]`는 `[y][x]`가 된다 ! 또한 대소관계도 다르다 !
   * 사영시켜서 생각하면 쉽다 !
   * 그냥 매번 사영된 i축, j축을 떠올리자.

<img src="../../fig/image-20201124142717480.png" alt="image-20201124142717480" style="zoom: 33%;" />

2. 우선순위를 둬라.
   * 일단 무조건 i좌표를 맞추고 j를 맞추자 ! 라던가 j를 먼저 맞추자! 라던가 하는 나만의 우선순위가 있어야 함
   * 2번 문제에서는 '바로 다음 행동'만 return하도록 시킴.
     1. 일단 i좌표를 같게 만든다.  => up / down만 명령
     2. 그리고 나서 j좌표에 대한 행동을 취한다. => left / right만 명령



##### 다른 사람의 풀이



##### 배워갈 점

* 좌표 다루는 법 !
* 헷갈릴 땐 역시 기능을 쪼개자.

```python
def displayPathtoPrincess(n,grid):
    
    def find_coordinates(target_alphabet):
        for ti in range(n):
            for tj in range(n):
                if grid[ti][tj] == target_alphabet:
                    return ti, tj
                
    pi, pj = find_coordinates('p')
    bi, bj = find_coordinates('m')
    
    moves = []
    
    def horizontal_direction(sj,ej):
        if sj == ej:
            return
        direction = 'RIGHT' if sj < ej else 'LEFT'
        moves.extend([direction]*abs(sj-ej))
    
    def vertical_direction(si,ei):
        if si == ei:
            return
        direction = 'DOWN' if si < ei else 'UP'
        moves.extend([direction]*abs(si-ei))

    horizontal_direction(bj,pj)
    vertical_direction(bi,pi)
    
    for move in moves:
        print(move)   
```



* `UP_LEFT, UP_right, down_left, down_right, left, up, down, right` 출력하기

```python
def where(start, end):
    si, sj = start
    ei, ej = end

    if si == ei:
        if sj > ej:
            return 'left'
        elif sj < ej:
            return 'right'
        return "i'm there!"

    elif si < ei:  # down
        if sj > ej:
            return 'down-left'
        elif sj < ej:
            return 'down-right'
        return "down"

    else:
        if sj > ej:
            return 'up-left'
        elif sj < ej:
            return 'up-right'
        return "up"
```

```python
(0, 0) up-left
(0, 1) up
(0, 2) up-right
(1, 0) left
(1, 1) i'm there!
(1, 2) right
(2, 0) down-left
(2, 1) down
(2, 2) down-right
```



* 바로 다음 action만 출력하고 싶을 때
  * 우선 순위 : i를 맞추자

```python
def move(start, end):
    si, sj = start
    ei, ej = end
    
    if si == ei:
        if sj < ej:
            return 'right'
        elif sj > ej:
            return 'left'
        return 'finish !'
    
    elif si < ei:
        return 'down'
    else:
        return 'up'
```

```python
while True:
    direction = move(start, end)

    if direction == 'finish !':
        break

    si, sj = start

    if direction == 'left':
        start = (si, sj-1)
    elif direction == 'right':
        start = (si, sj+1)
    elif direction == 'up':
        start = (si-1,sj)
    else:
        start = (si+1,sj)
```





# BotClean

> nxn grid위에 더러운 곳을 청소하기 위해 로봇이 움직인다.
>
> 이동 거리를 짧게, 그리고 모든 곳을 청소하기 위해서는 어떻게 움직여야 할까?



함수 형식 내 마음대로 바꿔더 되는건가?



## 1. 모든 점을 가장 작은 step으로 청소하는 방법



##### 다른 사람의 풀이

1. DFS로 모든 가능한 경로 만들고 그 중 가장 최적의 경로
   * 2번을 풀어보니까 1번의 의도는 이거인듯 !
   * 들려야 하는 모든 점을 안다면(미래를 안다면) 미리 다 행해보고 최선을 선택한다.
2. traveling salesman을 부르트포스



##### 내 풀이

> 시간 복잡도를 생각 안한 것, 나름 생각한 것 두개로 짰는데 후자 역시 시간복잡도를 그렇게 줄여주진 못했다. (3번을 통과 못하므로)



##### 1 ) 매 위치마다 가장 가까운 점을 골라서 그 점을 청소하도록 한다.

```python
#!/usr/bin/python

# Head ends here

import heapq


class Board:
    def __init__(self, grid):
        self.grid = grid
        self.bot = None
        self.dirty = []
        self.get_dirty_coordinates()

    def is_clean(self):
        return len(self.dirty) == 0

    def get_dirty_coordinates(self):
        for i in range(5):
            for j in range(5):
                if self.grid[i][j] == 'd':
                    self.dirty.append([i, j])

    def get_action(self, si, sj, ei, ej):
        if si == ei:
            if sj == ej:
                return 'CLEAN'
            elif sj < ej:
                return 'RIGHT'
            else:
                return 'LEFT'
        elif si < ei:
            return 'DOWN'
        else:
            return 'UP'

    def next_move(self):
        curri, currj = self.bot
        dist = []

        for di, dj in self.dirty:
            heapq.heappush(dist, (abs(di - curri) + abs(dj - currj),
                                  (di, dj)))

        targeti, targetj = dist[0][-1]

        return self.get_action(curri, currj, targeti, targetj)

    def execute(self, action):
        if action == 'CLEAN':
            self.dirty.remove(self.bot)
        elif action == 'LEFT':
            self.bot[1] -= 1
        elif action == 'RIGHT':
            self.bot[1] += 1
        elif action == 'UP':
            self.bot[0] -= 1
        else:
            self.bot[0] += 1

if __name__ == "__main__":
    pos = [int(i) for i in input().strip().split()]
    board = [[j for j in input().strip()] for i in range(5)]
    board_state = Board(board)
    board_state.bot = [pos[0], pos[1]]

    while not board_state.is_clean():
        action = board_state.next_move()
        print(action)
        board_state.execute(action)
```



##### 2 ) 1 step 의 범위 내에 있는 것 부터 처리 => 처리할 것이 없다면 2 step => 3 step => ...

BFS로 구현

```python
from collections import deque

class Board:
    def __init__(self, bot_pos, grid_dim, grid):
        self.grid = grid
        self.h, self.w = grid_dim
        self.bot = bot_pos
        self.dirty = []
        self.get_dirty_coordinates()

    def is_clean(self):
        return len(self.dirty) == 0

    def get_dirty_coordinates(self):
        for i in range(self.h):
            for j in range(self.w):
                if self.grid[i][j] == 'd':
                    self.dirty.append([i, j])

    def get_action(self, si, sj, ei, ej):
        if si == ei:
            if sj == ej:
                return 'CLEAN'
            elif sj < ej:
                return 'RIGHT'
            else:
                return 'LEFT'
        elif si < ei:
            return 'DOWN'
        else:
            return 'UP'

    def next_move(self):
        curri, currj = self.bot

        queue = deque([self.bot])
        visited = [[False]*self.w for _ in range(self.h)]

        while queue:
            x,y = queue.popleft()
            if self.grid[x][y] == 'd':
                return self.get_action(self.bot[0], self.bot[1],x,y)
            visited[x][y] = True

            for dx,dy in [(0,1),(1,0),(0,-1),(-1,0)]:
                if x+dx >= self.h or x+dx < 0 or y+dy < 0 or y+dy >= self.w:
                    continue
                if visited[x+dx][y+dy]:
                    continue
                queue.append((x+dx, y+dy))

    def execute(self, action):
        if action == 'CLEAN':
            self.grid[self.bot[0]][self.bot[1]] = '-'
            self.dirty.remove(self.bot)
        elif action == 'LEFT':
            self.bot[1] -= 1
        elif action == 'RIGHT':
            self.bot[1] += 1
        elif action == 'UP':
            self.bot[0] -= 1
        else:
            self.bot[0] += 1

if __name__ == "__main__":
    pos = [int(i) for i in input().strip().split()]
    dim = [int(i) for i in input().strip().split()]
    board = [[j for j in input().strip()] for i in range(5)]

    board_state = Board(pos,dim,board)
    board_state.bot = [pos[0], pos[1]]

    while not board_state.is_clean():
        action = board_state.next_move()
        print(board_state.bot, action)
        board_state.execute(action)
        for g in board_state.grid:
            print(*g)
```



##### 3 ) 구역을 나누고 차례로 들리면서 청소해나간다.







## 2. stochastic일 때

> 1번 clean이 끝나면 새로운 dirty가 생긴다.
>
> 200번의 움직임 안에 최대한 많은 dirty를 청소해라.



**내 풀이**

* 1번과 같다. 항상 1개의 dirty만 있으니까 항상 걔로 향해야한다.



## 3. Large ver

위에서 언급한 버전으로는 할 수 없다 !



**1 ) 1번-1,2 코드로 통과할 수 없는 이유 => 낭비가 생긴다.**

무조건 가까운 곳부터 다 방문한다면, 왼쪽 아래 끝에서 오른쪽 위 끝으로 가는 데에 8을 낭비하게 됨.

처음부터 오른쪽 위 끝을 들렸다가 다른 것을 처리하는게 훨씬 빠르다 !

```python
0 0
5 5
b---d
dd---
d-d--
dd---
d----
```

