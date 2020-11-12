[TOC]



**:pear: 새로 만든 node의 pointer부터 처리하면 헷갈릴 일이 없다 !**



# :black_flag: Linear/Ordered List 선형/순서 리스트

선형/순서 리스트 : 위의 목록에 순서(index)가 있는 리스트 ! ( 리스트 : 그저 나열할 목록들 )

> 정렬 되었다는 뜻이 아니라 값 사이에 순서가 생겼다는 뜻이다.

![image-20201112170501707](C:%5CUsers%5Chaeyu%5CDocuments%5CTypora%20%EB%AC%B8%EC%84%9C%5Cfig%5Cimage-20201112170501707.png)

![image-20201112203919098](C:%5CUsers%5Chaeyu%5CDocuments%5CTypora%20%EB%AC%B8%EC%84%9C%5Cfig%5Cimage-20201112203919098.png)



**메모리 저장 방식에 따라**

1. 순차 리스트 == 배열

   * def : 번호와 번호에 대응하는 데이터로 이루어진 자료구조. 번호 자체가 상대적인 위치가 된다.

   * **논리적 순서 == 물리적 순서**

   * 메모리에 **순서대로** 저장되어 있음. **인접한 곳**에 연속적으로 저장함.

   * 삽입, 삭제 연산이 일어나면 **저장 순서도 변경**됨.

     * 논리적 순서에 맞게 물리적 순서도 변경 (오버헤드 발생)

   * 랜덤 엑세스가 가능하다. 그러므로 특정 원소의 **위치를 알 수 있다.**

     > 뒤에서 나오지만 이 특성 덕분에 이진 검색이 가능한 것이고, 연결 리스트는 그렇지 않기 때문에 불가능하다.

2. 연결 리스트

   * **논리적 순서 != 물리적 순서**

   * 메모리에 **뿔뿔히 흩어져 저장**되어 있음.

   * 삽입, 삭제 연산이 일어나도 기존 위치는 **바뀌지 않음**.

   * 랜덤 엑세스 불가능. 그러므로 특정 원소의 위치를 알 수 없다.

     > 순서가 있는 것과 위치를 바로 찾을 수 있는 것은 다른 문제임을 헷갈리지 말자 !



# :black_circle: Array 배열



# :black_circle: Linked List 연결 리스트



**자기 참조 구조체**

자기 자신과 같은 타입의 구조체를 가르키는 pointer(link)를 속성으로 포함하는 구조체

node 내에 다른 node를 가르키는 pointer 속성이 존재한다 !



`new->link = p->link` 파이썬에서는 `new.link = p.link`

```python
class Node:
    def __init__(self, value)
    self.value = value
    self.pointer = None # 다음 Node obj의 주소를 가르키게 된다 !
```

```python
a = Node(10)
b = Node(20)
a.pointer = b # a의 pointer를 통해 a,b가 이어진다.
```



# :black_heart: **단순 연결 리스트 Singly Linked **

* 노드마다 link가 하나만 있는 리스트

* 첫번째 노드의 주소만 알고 있으면 된다.

  ```python
  start = a # start 즉 첫번째 노드 a만 가질 수 있으면 리스트 전체를 알 수 있다 !
  ```

* 마지막 노드의 link 포인터는 NULL



## :baguette_bread: 1 ) Chain

* 마지막 node의 pointer가 NULL인 경우



### **[length] 연결 리스트의 길이 계산**

> self.pointer가 None이 되면 리스트가 끝난다.

```python
a,b,c,d = Node(10), Node(20), Node(30), Node(40)
a.pointer = b
b.pointer = c
c.pointer = d
```

```python
curr_node = start
length = 1

while True:
    if curr_node.pointer is not None:
        curr_node = curr_node.pointer
        length += 1
    else:
        break

print('길이는 :',length)
```



### **[insert] 연결 리스트의 노드 추가**

> :raising_hand_woman: [실수 포인트] **새로 추가되는 노드**의 링크를 **먼저** 바꾸자 !

`Q` : 20 다음 노드에 25를 삽입해보자.

1. 20의 위치를 알아낸다.
2. 20의 pointer를 25로, 25의 pointer를 30으로 변경한다.



```python
def find(value):
    curr_node = start
    while True:
        if curr_node.pointer is None:
            return False # 찾으려는 value가 없다.
        if curr_node.value == value:
            return curr_node
        curr_node = curr_node.pointer
        
def insert(prev_value, target_value):
    prev_node = find(prev_value)
    if not prev_node:
        return False
    target_node = Node(target_value)
    
    ######################################## << 순서에 주의하자 >> target부터 pointer를 이어받아야 함 !
    
    # 2. prev의 pointer를 target이 가져가고
    target_node.pointer = prev_node.pointer
    
    # 3. prev의 pointer를 target으로 수정한다.
    prev_node.pointer = target_node
    
    ########################################
    
    return True
```



### [delete] 연결 리스트의 노드 삭제

1. 삭제할 node의 **이전 노드** 의 pointer를 찾아서

   > target value의 이전 노드를 어떻게 찾을 것인가 !

2. 삭제할 node의 pointer로 바꾸면 끝 !

```python
def find_prev(value):
    curr_node = start
    prev_node = None
    
    while True:
        if curr_node.pointer is None:
            return False
        if curr_node.pointer.value == value:
            return curr_node
        prev_node, curr_node = curr_node, curr_node.pointer

def delete(value):
    global start
    # head node를 삭제하려고 한다면 start가 head의 다음 node를 가르키도록 해야한다.
    # (길이 1에서 삭제하는 경우는 예외처리 하지 않았음.)
    
    if start.value == value:
        start = start.pointer
        return True
    
    prev_node = find_prev(value)
    target_node = find(value)
   
    prev_node.pointer = target_node.pointer
    
    return True
```



삽입/삭제시에 생성된 **Node obj의 주소는 변하지 않음 !**

그저 속성인 `pointer`만 조작해주었을 뿐 !!!!



### [inverse] 연결 리스트 뒤집기

```python
prev, curr = None, start

while True:
    if temp is None:
        start = prev
        break
    temp = curr.pointer
    curr.pointer = prev
    prev, curr = curr, temp
```



### [concat] 두 체인을 통합하기

![image-20201112193111865](C:%5CUsers%5Chaeyu%5CDocuments%5CTypora%20%EB%AC%B8%EC%84%9C%5Cfig%5Cimage-20201112193111865.png)

* A 또는 B가 NULL이면 나머지를 바로 return

```python
1. A에서 시작해서 pointer가 NULL인 node를 찾는다.
2. node의 pointer를 B로 준다.
```



---



### 배열 v.s. 연결 리스트

#### 1. 저장 방식의 차이

`배열 ; int ls[8]` 다음 데이터에 대한 **주소를 알 필요는 없음**. 메모리의 **인접한 곳에 저장**

* ls[2]를 저장하기 위해 ls[3]의 주소를 알 필요가 없다.

`연결 리스트 ; struct node` : 필요할 때마다 메모리를 **동적으로 할당** ! 그러므로 메모리의 이곳 저곳에 흩어져있음.

pointer를 통해 다음 node의 주소를 알고있어야함.



#### 2. 메모리 사용 측면

배열의 length를 미리 안다면, 배열이 효과적이다.

* 다음 노드의 **주소를 기억할 필요가 없으니까!**
* 연결 리스트는 다음 노드의 주소만큼 더 저장해야 함.



그러나 데이터의 수를 모를 경우 연결 리스트가 유리하다.

> 배열도 동적할당이 가능하지만 어쨌든 더 키우기 위해 다시 할당이 필요

* 필요할 때마다 **동적으로 할당**할 수 있으니까.



#### 3. 정렬된 데이터의 순서 유지

배열과 링크드 리스트 모두 순서 리스트/선형 리스트임. (ordred/linear) .

배열의 경우 삽입/삭제가 일어나면 나머지 원소의 index를 다시 싹 변경

그러나 연결 리스트는 위에서도 말했듯이 기존 원소는 처음 할당된 주소에 그대로 있음.

즉 ! 배열은 연산에 따라 계속 주소를 조정하지만 연결 리스트는 그렇지 않기 때문에 **어느 값이 중간에 있는지** 알 수 없음.



:raising_hand: **연결 리스트는 ==이진 검색이 불가능==하다.**

* 중간 노드의 주소를 알 수 없으니까 ! 바로 뙇 이동할 수 없음.

  > length 계산하고 length//2로 이동하면 되는거 아니야? 했는데 **랜덤 엑세스**가 안되므로 불가능하다 !
  >
  > * 연결 리스트는 length//2로 가기 위해 O(N)이 소요
  >   * 중간 주소인 length//2로 바로 이동 불가 ! 순회해야함
  > * Array는 O(1)
  >
  > **O(logN)으로 검색 불가 !**



### Quiz. 연결 리스트로 구현하는 것이 배열보다 효율적인 연산은?

연결 리스트로 구현하는 것이 효율적인 연산은? (길이는 100이라고 하자)

1. 리스트의 90번째 위치에 있는 데이터를 출력
2. 리스트의 2번째 위치에 있는 데이터를 삭제
3. 리스트에서 0보다 큰 데이터를 모두 출력
4. 리스트의 50번째 데이터를 다른 값으로 변경

답은 2번 ! 연결 리스트는 랜덤 엑세스가 불가능하므로 1,4를 하기 위해 90번/50번씩 pointer를 따라 이동해야함. array는 O(1)

0보다 큰 데이터 출력은 모두 O(N)으로 다 훑어야 할듯 !



### 연결리스트로 stack, queue 구현하기

* 배열로 구현하는 경우 발생하는 문제
  * 메모리 낭비
  * size가 고정되어 있으므로 stack FULL발생 가능성

stack은 self.top, queue는 self.front, self.rear로 관리

> 원형 연결 리스트 + 마지막을 이름으로 설정 => Queue 구현시 front/rear 둘 다 쓸 필요 없고 A로 운용 가능 !



queue는 front에서 나가고 rear 로 들어온다 (거꾸로 생각할뻔 ㅠㅠ)

pointer는 다 오른쪽으로 향한다.

==**(삭제)**== <= `front, 1, 2, 3,4 .. rear` <= **==(삽입)==**

1. 삽입 : target_node생성, rear의 pointer를 target_node의 pointer로, rear를 target_node로
2. 삭제 : front를 front의 pointer로 바꿔준다.



stack, queue의 삽입, 삭제 연산은 모두 O(1) ! deque도 마찬가지일까?

### Quiz. Deque를 단순 연결 리스트로 구현할 때 O(1)에 불가능한 연산은?

1. insertFirst
2. insertLast
3. deleteFirst
4. DeleteLast



정답은 4번 !

마지막 값을 지우려면

1. rear의 prev에 있는 node를 찾아서

   > node.pointer.value == rear.value일 때까지 탐색. O(N)소요

2. prev.pointer = NULL

3. rear = prev



### 연결 리스트를 이용한 다항식 구현

`class poly` 를 이용해서 생성된 두 poly a,b에 대해 c를 만드는 방법

`class node`는 `value, expon, pointer` 가짐

언제나 뒤에 입력되므로 queue로 구현

```python
a의 모든 node를 돌면서:
    b의 현재 노드와 a의 현재 노드의 지수를 비교.
    if 지수가 같다면:
        c의 뒤에 insert(지수, 두 value의 합)
        a,b의 현재 노드를 pointer를 통해 변경
    elif a의 지수가 크다면:
        c.insert(a의 지수, a의 value)
        a의 현재노드 변경(pointer)
    else:
        b로 동일 연산
```



---



## :doughnut: 2 ) **원형 연결 리스트 Circular Linked**

* 마지막 노드와 첫 노드를 연결시킴. `last.pointer = 맨 앞 !`

* NULL을 없애고 싶어서 원형을 만든건데 결국 원소가 없을 때에는 `self.head = NULL`이 필연적으로 발생

  => **==아무것도 들어있지 않은 `head_node`를 만들어놓자 !==** 



**head_node 덕분에 NULL검사를 할 필요가 없으므로 다항식 더하기를 훨씬 효율적으로 할 수 있다.**

* head node를 표시하기 위해서 ` node.expon = -1`로 주자. 이는 연산을 더 쉽게 하도록 도와주기도 한다.

* 단순 연결을 이용하면 a 혹은 b가 끝났는지 아닌지도 검사해주어야 한다.
  둘 중 **먼저 끝난 다항식이 -1을** 가르키게 되면, **당연히 나머지의 지수가 크기 때문에** 예외처리 없이도 **자연스럽게 모두 더해지고**
  **모두 -1이 되는 경우에 연산을 중지**하면 된다 !

  ![image-20201112185711814](C:%5CUsers%5Chaeyu%5CDocuments%5CTypora%20%EB%AC%B8%EC%84%9C%5Cfig%5Cimage-20201112185711814.png)





### Quiz. 원형 연결의 sum을 계산할 때 head는 따로 처리가 필요하다.

그림은 저래도 A는 `Node(10)`을 가르키고 있다는 것을 잊지 말자 .. `A.link`는 `Node(20)`이다.

![image-20201112190739311](C:%5CUsers%5Chaeyu%5CDocuments%5CTypora%20%EB%AC%B8%EC%84%9C%5Cfig%5Cimage-20201112190739311.png)



### [length] 원형 연결 리스트 순회하기

* 단순 연결이라면 pointer가 NULL인 경우까지 가면 되는데, 원형이면 어떻게 할까?
* 처음 노드로 돌아왔을 때 반복문을 끝내자.
* **==단 첫 노드는 따로 처리해야한다==**



```python
print(start.value, end = ' => ')
curr_node = start.pointer

while True:
    ###################### 처음 노드로 올 때 중지할거니까, 시작은 따로 처리해야 한다
    if curr_node == start:
        break
    ######################
    print(curr_node.value, end = ' => ')
    curr_node = curr_node.pointer
```



### **원형 연결 리스트의 이름은 ==맨 뒤 node==로 하는 것이 좋다.**

맨앞 삽입, 맨뒤 삽입 연산이 둘다 O(1)만에 가능하기 때문 !

![image-20201112201113904](C:%5CUsers%5Chaeyu%5CDocuments%5CTypora%20%EB%AC%B8%EC%84%9C%5Cfig%5Cimage-20201112201113904.png)



### Queue를 마지막이 이름인 원형 연결 리스트로 구현한다면 front, rear 중 하나만 있으면 된다!

A 자체가 rear, `A.pointer`가 front가 되므로!



### Quiz. 두 연결 리스트의 중간에 있는 원소 swap하기

> 내가 짜서 정답을 찾기보단 하나씩 해보는게 더 빠름 !

<img src="C:%5CUsers%5Chaeyu%5CDocuments%5CTypora%20%EB%AC%B8%EC%84%9C%5Cfig%5Cimage-20201112202209927.png" alt="image-20201112202209927" style="zoom:67%;" />

<img src="C:%5CUsers%5Chaeyu%5CDocuments%5CTypora%20%EB%AC%B8%EC%84%9C%5Cfig%5Cimage-20201112202230685.png" alt="image-20201112202230685" style="zoom:45%;" />

### Quiz. 원형 리스트로 Queue 구현

![image-20201112202538559](C:%5CUsers%5Chaeyu%5CDocuments%5CTypora%20%EB%AC%B8%EC%84%9C%5Cfig%5Cimage-20201112202538559.png)

<img src="C:%5CUsers%5Chaeyu%5CDocuments%5CTypora%20%EB%AC%B8%EC%84%9C%5Cfig%5Cimage-20201112202522532.png" alt="image-20201112202522532" style="zoom:50%;" />





# :black_heart: **이중 연결 리스트 Doubly Linked**

* 양방향으로 가르킬 수 있는 연결 리스트
* `left_link`, `right_link`를 모두 가진다.
* **삭제를 위해 이전 노드를 다시 탐색할 필요가 없게 된다 !**

```python
class Node:
    def __init__(self,value):
        self.value = value
        self.left_point = None
        self.right_point = None
```



## :baguette_bread: 1 ) chain

모야 .. 헷갈려

언제는 new node부터 연결하라며 ...

http://www.kmooc.kr/courses/course-v1:YeungnamUnivK+YU216002+2018_01/courseware/4bdc80eb4a7c4608a6bbc7dae63883cc/85fa3c2b2c354c62a7ac62dd48ce93b0/?child=first

<img src="fig/image-20201112223558846.png" alt="image-20201112223558846" style="zoom:67%;" />



## :doughnut: 2 ) 원형







# Reference

* [KMOOC 자료구조 강의](http://www.kmooc.kr/courses/course-v1:YeungnamUnivK+YU216002+2018_01/courseware/8aa77078c0694400903bc296553c633f/6b097b38751d4c1cb0ac626f291f242a/1?activate_block_id=block-v1%3AYeungnamUnivK%2BYU216002%2B2018_01%2Btype%40vertical%2Bblock%40d3417ed137a742eba8c860c5ae96afb7)
