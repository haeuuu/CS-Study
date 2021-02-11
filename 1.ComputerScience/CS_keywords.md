# resource 자원

In [computing](https://en.wikipedia.org/wiki/Computing), a **system resource**, or simply **resource**, is any physical or virtual component of limited availability within a computer system. Every device connected to a computer system is a resource. Every internal system component is a resource.



# :yellow_heart: process

== program, 작업

컴퓨터에서 연속적으로 실행되고 있는 컴퓨터 프로그램.

스케줄링의 대상이 되는 "작업 task"라는 용어와 같은 의미로 쓰인다.

In computing, a **process** is the [instance](https://en.wikipedia.org/wiki/Instance_(computer_science)) of a [computer program](https://en.wikipedia.org/wiki/Computer_program) that is being executed by one or many threads. It contains the program code and its activity.



## program

하드 디스크 등에 저장되어있는 **실행 코드**

=> 프로세스는 얘를 **실행시킨 자체** 또는 이 프로그램이 **메모리 상에서 실행하는 작업의 단위**를 말한다.

ex ) 하나의 program을 여러 번 구동하면?? 여러개의 process가 메모리상에서 실행된다.



## multiprocessing

여러개의 프로세서를 사용하는 것

쉽게 생각하면 여러개의 cpu를 사용하는 것.



## multi tasking

같은 시간에 여러개의 프로그램을 띄우는 시분할(Time sharing) 방식 !!!!



# :yellow_heart: thread

스레드는 프로세스의 component !

하나의 process 내에서 실행되는 작업흐름의 단위.

여러개의 thread는 하나의 프로세스에 존재할 수 있다. 

a **thread** of execution is the smallest sequence of programmed instructions that can be managed independently by a [scheduler](https://en.wikipedia.org/wiki/Scheduling_(computing)), which is typically a part of the [operating system](https://en.wikipedia.org/wiki/Operating_system).



## 데몬 Daemon

사용자가 직접 제어하지 않고 백그라운드에서 돌면서 여러 작업을 하는 프로그램

백그라운드에서 상주하다가 사용자가 요청하면 즉시 대응하는 프로세스

> 아 그래서 queue에 넣을때까지 기다리라고 말하기 위해 daemon = True를 했나보다 ....





# :yellow_heart: 비동기 Asynchronous 프로그래밍

병렬 처리 방식.

결과를 기다리는 부분 없이 바로 다음 작업을 실행할 수 있게 된다. (소포 맡기고 전달 되던 말던 나는 집에가서 내할일 한다.)

**Asynchrony**, in [computer programming](https://en.wikipedia.org/wiki/Computer_programming), refers to the occurrence of events independent of the main [program flow](https://en.wikipedia.org/wiki/Control_flow) and ways to deal with such events

* "outside" events such as the arrival of [signals](https://en.wikipedia.org/wiki/Unix_signal), 
*  without the program *blocking* to wait for results



## 동시성 프로그래밍 concurrently

같은 종류의 작업이 가능한 많이 동시에 일어나기를 바란다. => ==thread와 관련!!!==

ex : 다수 클라이언트의 요청에 응답하는 서버 만들기



## 병렬 프로그래밍 parallel

어떤 하나의 계산을 쪼개서 병렬적으로 수행한 후 빠르게 끝내길 원한다.





# :yellow_heart: kernel

![img](https://upload.wikimedia.org/wikipedia/commons/thumb/8/8f/Kernel_Layout.svg/200px-Kernel_Layout.svg.png)

운영 체제의 핵심을 이루는 요소. 소스 코드

The **kernel** is a [computer program](https://en.wikipedia.org/wiki/Computer_program) at the core of a computer's [operating system](https://en.wikipedia.org/wiki/Operating_system) with complete control over everything in the system

It is the "portion of the operating system code that is always resident in memory

**컴퓨터 내의 자원**을 **사용자** 프로그램 user application이 **사용할 수 있도록 관리**해주는 프로그램

응용 프로그램의 수행에 필요한 서비스를 제공한다.

프로세스, 메모리, 네트워크, 파일 시스템의 관리를 맡는다.



# :yellow_heart:scheduling 스케줄링

어떤 work가 어떤 resource에 포함될지를 결정하는 것.

In [computing](https://en.wikipedia.org/wiki/Computing), **scheduling** is the method by which work is assigned to resources that complete the work



## 문맥 교환 context switching

새로운 프로세스에게 cpu를 할당하기 위해 <u>현재 cpu 상태를 저장</u>하고 할당해주는 작업 => ***하나의 cpu로 이것저것 돌릴 때 필요하겠군 !***

overhead 발생 요인 중 하나

resume ; 재개

In [computing](https://en.wikipedia.org/wiki/Computing), a **context switch** is the process of storing the state of a [process](https://en.wikipedia.org/wiki/Process_(computing)) or [thread](https://en.wikipedia.org/wiki/Thread_(computing)), so that it can be restored and resume [execution](https://en.wikipedia.org/wiki/Execution_(computing)) at a later point. This allows multiple processes to share a single [central processing unit](https://en.wikipedia.org/wiki/Central_processing_unit) (CPU), and is an essential feature of a [multitasking operating system](https://en.wikipedia.org/wiki/Multitasking_operating_system).



The precise meaning of the phrase “context switch” varies. In a multitasking context, it refers to the process of storing the system state for one task, so that task can be paused and another task resumed. A context switch can also occur as the result of an [interrupt](https://en.wikipedia.org/wiki/Interrupt), such as when a task needs to access [disk storage](https://en.wikipedia.org/wiki/Disk_storage), freeing up CPU time for other tasks.



## 선점(나중에 왔어도 뺏을 수 있다) / 비선점(그런거 없음 순서대로)

헷갈림 선점이 뺏을 수 있는거임 !!! 선점 == preemptive

선점은 새로 들어오는 task의 관점에서 쓰였다.

* 굴러들어온 돌이 선점할 수 있으면 선점 ! 못하면 비선점 !



## 선점

In [computing](https://en.wikipedia.org/wiki/Computing), **preemption** is the act of temporarily interrupting a [task](https://en.wikipedia.org/wiki/Task_(computing)) being carried out by a [computer system](https://en.wikipedia.org/wiki/Computer), without requiring its cooperation, and with the intention of resuming the task at a later time. 

**선점 스케줄링**(preemptive scheduling)은 [시분할 시스템](https://ko.wikipedia.org/wiki/시분할_시스템)에서 [타임 슬라이스](https://ko.wikipedia.org/wiki/타임_슬라이스)가 [소진](https://ko.wikipedia.org/wiki/소진)되었거나, [인터럽트](https://ko.wikipedia.org/wiki/인터럽트)나 [시스템 호출](https://ko.wikipedia.org/wiki/시스템_호출) 종료 시에 더 높은 우선 순위 [프로세스](https://ko.wikipedia.org/wiki/프로세스)가 발생 되었음을 알았을 때, 현 실행 프로세스로부터 강제로 [CPU](https://ko.wikipedia.org/wiki/중앙_처리_장치)를 회수하는 것을 말한다.



* 라운드 로빈
  * <u>우선순위를 두지 않고</u>, 시간 단위 (time quantum)으로 cpu를 할당하는 방법.
  * 10초동안 실행하고 다음 process, 다시 또 10초동안 실행하고 다음 process ... (우선순위는 맨 끝으로)
  * overhead는 크지만 응답 시간은 짧아진다.
  * **Round-robin** (RR) is one of the algorithms employed by [process](https://en.wikipedia.org/wiki/Process_scheduler) and [network schedulers](https://en.wikipedia.org/wiki/Network_scheduler) in [computing](https://en.wikipedia.org/wiki/Computing).[[1\]](https://en.wikipedia.org/wiki/Round-robin_scheduling#cite_note-ostep-1-1)[[2\]](https://en.wikipedia.org/wiki/Round-robin_scheduling#cite_note-Zander-2) As the term is generally used, [time slices](https://en.wikipedia.org/wiki/Preemption_(computing)#Time_slice) (also known as time quanta)[[3\]](https://en.wikipedia.org/wiki/Round-robin_scheduling#cite_note-3) are assigned to each process in equal portions and in circular order, handling all processes without [priority](https://en.wiktionary.org/wiki/priority)
* SRT (shortest **remaining** time. 최소 잔류 시간 우선)
  * 비선점의 shortest job first의 선점 버전.
  * 실행중인 task의 남은 시간과 이제 실행할 task들의 실행 시간을 비교하여 더 짧은 실행시간을 쓴다.
  * **shortest remaining time first (SRTF)**, is a [scheduling](https://en.wikipedia.org/wiki/Scheduling_(computing)) method <u>that is a [preemptive](https://en.wikipedia.org/wiki/Preemption_(computing)) version of [shortest job next](https://en.wikipedia.org/wiki/Shortest_job_next) scheduling.</u> In this scheduling algorithm, the [process](https://en.wikipedia.org/wiki/Process_(computing)) with the smallest amount of time remaining until completion is selected to execute. 
* 선점 우선순위
* 다단계 큐
  * task를 그룹으로 묶고(영구적으로), 또 그룹마다의 queue가 생기고 ...
  * **Multi-level queueing**, used at least since the late 1950s/early 1960s, is a queue with a predefined number of levels. Unlike the [multilevel feedback queue](https://en.wikipedia.org/wiki/Multilevel_feedback_queue), items get assigned to a particular level at insert (using some predefined algorithm), and thus cannot be moved to another level.
  * **Multi-level queue** [[1\]](https://en.wikipedia.org/wiki/Multilevel_queue#cite_note-osc-1):196 scheduling algorithm is used in scenarios where the processes can be classified into groups based on property like process type
  * q1에 있던 친구들이 다 끝나야 q2가 돌아감. 이 때 q1에 있던 친구가 끝나는 순간부터 q2의 첫번째 task가 시작되는데, 만약 얘가 돌아가던 중 q1에 새로운 task가 도착하더라도 일단 q2의 1은 끝나야 q1의 1이 돌아간다.
    ![image-20200925130339163](fig/image-20200925130339163.png)
* 다단계 피드백 큐
  * 큐 사이에서 task들이 이동할 수 있다.



## 비선점

* 빨리 들어온거부터 처리 FCFS (처음 come, 처음 serve) FIFO

  * 선입 선처리

* SJF(Shortest job first)

  *  is a [scheduling policy](https://en.wikipedia.org/wiki/Scheduling_algorithm) that selects for execution the waiting [process](https://en.wikipedia.org/wiki/Process_(computing)) with the smallest execution time.[[1\]](https://en.wikipedia.org/wiki/Shortest_job_next#cite_note-ostep-1-1) SJN is a non-[preemptive](https://en.wikipedia.org/wiki/Preemption_(computing)) algorithm.
  * 평균 대기 시간을 줄여준다.
    * Shortest job next is advantageous because of its simplicity and because it minimizes the average amount of time each process has to wait until its execution is complete
  * 그러나 긴 시간이 필요한 process의 대기시간이 무한정 길어지는 문제가 있다. 

* HRRN(Highest response ratio next) 최상 응답 비율 순서

  > SJF + aging (에이징은 오직 대기시간만 고려)

  * 일단 SJF의 순서에 의해 시행하되, 너무 오래 기다리지 않도록 해준다.
  * **Highest response ratio next** (**HRRN**) [scheduling](https://en.wikipedia.org/wiki/Scheduling_(computing)) is a [non-preemptive discipline](https://en.wikipedia.org/w/index.php?title=Non-preemptive_discipline&action=edit&redlink=1). It was developed by [Brinch Hansen](https://en.wikipedia.org/wiki/Per_Brinch_Hansen) as modification of [shortest job next](https://en.wikipedia.org/wiki/Shortest_job_next) (SJN) to mitigate the problem of [process starvation](https://en.wikipedia.org/wiki/Process_starvation).
  * 1 + 대기시간/실행시간 => 실행 시간보다 대기 시간이 너무 길어지면 우선순위가 커진다.

* 기한부
  * 일정 시간 안에 끝내도록 만든다.
  * 일정 시간 안에 끝내지 못했다면 처음부터 다시 실행시킴 (라운드 로빈이랑은 다름.)



##### Aging 에이징

* 오직 waiting time에 기반하여 우선순위를 키워준다
* 그냥 우선순위큐를 이용하면 process starvation 프로세스 기근 현상이 발생한다.
* In [Operating systems](https://en.wikipedia.org/wiki/Operating_system), **aging** (US English) or **ageing** is a [scheduling](https://en.wikipedia.org/wiki/Scheduling_(computing)) technique used to avoid [starvation](https://en.wikipedia.org/wiki/Resource_starvation)



# 1. starvation 기아 상태

프로세스가 계속해서 자원을 가져오지 못하는 상황. 과도한 스케줄링에 의해 이런 상태가 발생할 수 있다.

프로세스가 영원히 denied 되는 현상.

In [computer science](https://en.wikipedia.org/wiki/Computer_science), **resource starvation** is a problem encountered in [concurrent computing](https://en.wikipedia.org/wiki/Concurrent_computing) where a [process](https://en.wikipedia.org/wiki/Computer_process) is perpetually denied necessary [resources](https://en.wikipedia.org/wiki/Resource_(computer_science)) to process its work.[[1\]](https://en.wikipedia.org/wiki/Starvation_(computer_science)#cite_note-1) 



# 2. deadlock 교착 상태

두 개 이상의 작업이 <u>서로를 무한정 기다리는</u> 현상

ex ) 외나무다리에서 서로 비키기를 무한히 기다려서 결국 아무도 건너지 못한다.

In [concurrent computing](https://en.wikipedia.org/wiki/Concurrent_computing), a **deadlock** is a state in which each member of a group is waiting for another member,

In an [operating system](https://en.wikipedia.org/wiki/Operating_system), a deadlock occurs when a [process](https://en.wikipedia.org/wiki/Process_(computing)) or [thread](https://en.wikipedia.org/wiki/Thread_(computing)) enters a waiting [state](https://en.wikipedia.org/wiki/Process_state) because a requested [system resource](https://en.wikipedia.org/wiki/System_resource) is held by another waiting process, which in turn is waiting for another resource held by another waiting process.



## 교착 상태가 일어나게 되는 조건

<img src="fig/image-20200925134822063.png" alt="image-20200925134822063" style="zoom:33%;" />

1. **상호배제** : 한 번에 한 개의 프로세스만이 공유 자원을 사용할 수 있다.

   > 특정 도로 위치에서는 하나의 자동차만 있을 수 있다.

   * **남 안주고 나만 쓸거야 !!!!**
   * 두개의 process가 있더라도 동시에 두 process가 돌아갈 수 없음.

2. **점유 대기** : 할당된 자원을 <u>가진 상태</u>에서 다른 자원을 기다림.

   > 앞으로 한 칸 더 가려고 한다.

   * **나 지금 process 있긴 한데, 하나 더 필요해 내놔 !!!!**

3. 비선점 : 한 프로세스가 끝나기 전까지는 다른 일을 할 수 없음

   > 누가 앞으로 한 칸 가기 전까지는 내가 갈 수 없다.

4. 순환 대기

   > 누구 하나 양보하지 않고 너가 앞으로 가 ! 라고만 이야기를 하고 있다.

   * 일직선이 아니라 원형으로 서로가 서로에게 요구만 함. 끝나지가 않음.
   * 앞에있는 너도 자원 주고 뒤에있는 너도 자원 좀 줘봐 !



## 교착 상태 해결

1. 위 넷 중 하나의 조건을 깬다.
   1. 미리 모든 자원을 할당해주고, 자원이 없는 경우에만 요구하도록 한다.
   2. 이미 자원이 있는데 또 요구한다면, 반납하고 요구하도록 한다.
   3. 선형 순서로 대기시키고 한 방향으로만 자원을 요구하도록 한다.
2. 회피 (은행원 알고리즘)
   * 





# overhead 오버헤드

어떤 일을 처리하기 위해 들어가는 **간접적인 시간이나 메모리**

In [computer science](https://en.wikipedia.org/wiki/Computer_science), **overhead** is any combination of excess or <u>indirect computation time</u>, memory, bandwidth, or other resources that are required to perform a specific [task](https://en.wikipedia.org/wiki/Task_(computing))



# batch processing 일괄 처리

multiple items at once

일괄 처리 시스템은 일정 기간마다 주기적으로 한꺼번에 처리할 필요가 있고, 그룹별로 분류시킬 수 있는 성질을 가지고 있으며, 순차 접근방법을 사용할 수 있는 업무. 즉, 처리 요건이 일괄적인 업무에 대해 유사한 자료를 한데 모아 일정한 형식으로 분류한 뒤 한번에 일괄 처리함으로써 시간과 비용을 절감하여 업무의 효율성을 향상시킨다.

- [Batch processing](https://en.wikipedia.org/wiki/Batch_processing), the execution of a series of programs on a computer without human interaction



# Assembly 어셈블리어

기계어 코드





# Dynamic programming

### 1. overlapping subproblem

* 하나의 문제가 여러개의 부분으로 쪼개질 수 있다.
* 부분 문제의 덮개로 다 overlapping할 수 있다고 생각하자 ㅎㅎ

### 2. optimal substructure

* 전체 문제의 해결책이 부분 문제의 해결책으로 구해질 수 있을 때

 it refers to simplifying a complicated problem by breaking it down into simpler sub-problems in a [recursive](https://en.wikipedia.org/wiki/Recursion) manner

if a problem can be solved optimally by breaking it into sub-problems and then recursively finding the optimal solutions to the sub-problems, then it is said to have [optimal substructure](https://en.wikipedia.org/wiki/Optimal_substructure).





# :yellow_heart: OOP 객체지향의 4대 원칙

1. 추상화
   * 불필요한 것은 removing하고 객체의 속성 중 가장 중요한 것을 간추려내는 것.
   * In [software engineering](https://en.wikipedia.org/wiki/Software_engineering) and [computer science](https://en.wikipedia.org/wiki/Computer_science), **abstraction** is:
     - the process of removing physical, spatial, or temporal details[[2\]](https://en.wikipedia.org/wiki/Abstraction_(computer_science)#cite_note-:1-2) or [attributes](https://en.wikipedia.org/wiki/Attribute_(computing)) in the study of objects or [systems](https://en.wikipedia.org/wiki/System) to focus attention on details of greater importance;[[3\]](https://en.wikipedia.org/wiki/Abstraction_(computer_science)#cite_note-:0-3) it is similar in nature to the process of [generalization](https://en.wikipedia.org/wiki/Generalization);
2. 캡슐화
   * 변수와 함수를 하나로 묶어서 외부에서의 접근을 막는다.
   * 함수를 통해서만 접근할 수 있다.
   * In [object-oriented programming](https://en.wikipedia.org/wiki/Object-oriented_programming) (OOP), **encapsulation** refers to the bundling of data with the methods that operate on that data, or the restricting of direct access to some of an object's components.[[1\]](https://en.wikipedia.org/wiki/Encapsulation_(computer_programming)#cite_note-Rogers01-1) Encapsulation is used to hide the values or state of a structured data object inside a [class](https://en.wikipedia.org/wiki/Class_(computer_programming)), preventing unauthorized parties' direct access to them
   * 정보 은닉
     * 캡슐화에서 가장 중요한 개념. 다른 객체에게는 자신의 정보가 숨겨진다.
     * 각 객체의 수정이 다른 객체에는 고려되지 않는다.
3. 상속성
   * 상위 개념의 특징을 하위가 물려받는다
   * In [object-oriented programming](https://en.wikipedia.org/wiki/Object-oriented_programming), **inheritance** is the mechanism of basing an [object](https://en.wikipedia.org/wiki/Object_(computer_science)) or [class](https://en.wikipedia.org/wiki/Class_(computer_programming)) upon another object ([prototype-based inheritance](https://en.wikipedia.org/wiki/Prototype-based_programming)) or class ([class-based inheritance](https://en.wikipedia.org/wiki/Class-based_programming)), retaining similar implementation. Also defined as deriving new classes ([sub classes](https://en.wikipedia.org/wiki/Inheritance_(object-oriented_programming)#Subclasses_and_superclasses)) from existing ones such as super class or [base class](https://en.wikipedia.org/wiki/Fragile_base_class) and then forming them into a hierarchy of classes.
4. 다형성
   * 상위 클래스의 가상함수를 하위 클래스가 overriding하여 사용한다.
   * In [programming languages](https://en.wikipedia.org/wiki/Programming_language) and [type theory](https://en.wikipedia.org/wiki/Type_theory), **polymorphism** is the provision of a single [interface](https://en.wikipedia.org/wiki/Interface_(computing)) to entities of different [types](https://en.wikipedia.org/wiki/Data_type)[[1\]](https://en.wikipedia.org/wiki/Polymorphism_(computer_science)#cite_note-1) or the use of a single symbol to represent multiple different types.[[2\]](https://en.wikipedia.org/wiki/Polymorphism_(computer_science)#cite_note-Luca-2)



## :blue_heart: 다형성 polymorphsim 폴리모피즘

> 객체 종류와 상관없는 추상도 높은 method를 구현할 수 있다.
> <img src="fig/image-20200925111401472.png" alt="image-20200925111401472" style="zoom:33%;" />

같은 모양의 코드가 다른 동작을 하는 것

=> class에서 오버라이딩을 생각해보면 자식 class가 부모의 method를 수정하고, 같은 method임에도 다른 연산을 하니까 얘도 다형성

In [programming languages](https://en.wikipedia.org/wiki/Programming_language) and [type theory](https://en.wikipedia.org/wiki/Type_theory), **polymorphism** is the provision of a single [interface](https://en.wikipedia.org/wiki/Interface_(computing)) to entities of different [types](https://en.wikipedia.org/wiki/Data_type)[[1\]](https://en.wikipedia.org/wiki/Polymorphism_(computer_science)#cite_note-1) or the use of a single symbol to represent multiple different types.[[2\]](https://en.wikipedia.org/wiki/Polymorphism_(computer_science)#cite_note-Luca-2)

![image-20200925110458506](fig/image-20200925110458506.png)

## :blue_heart: Overloading 오버로딩

함수 명은 같은데 parameter에 따라 다른 실행을 하게되는 것.



In some [programming languages](https://en.wikipedia.org/wiki/Programming_language), **function overloading** or **method overloading** is the ability to create multiple [functions](https://en.wikipedia.org/wiki/Subprogram) of the same name with different implementations



##### example : 내가 정의한 class인 Num에 add 연산을 수행하고 싶을 때

```python
class Num:
    def __init__(self, num):
        self.num = num

mynum = Num(5)
mynum + 100 # error
```

```python
class Num:
    def __init__(self, num):
        self.num = num
        
    def __add__(self, query):
        return self.num + query
    
    def __radd__(self, query):
        return self.num + query

mynum = Num(5)
mynum + 100 # 105 return
```

```python
# __radd__가 왜필요해?
mynum + 100은 mynum.__add__(100) 을 호출하는 것과 같고, 이는 위 정의에 의해 105를 return
그러나 100 + mynum은 mynum.__radd__(100)을 호출하게 되므로 다시 obj와 int간의 연산 불가 error가 호출된다.
```



In [computer programming](https://en.wikipedia.org/wiki/Computer_programming), **operator overloading**, sometimes termed *operator [ad hoc polymorphism](https://en.wikipedia.org/wiki/Ad_hoc_polymorphism)*, is a specific case of [polymorphism](https://en.wikipedia.org/wiki/Polymorphism_(computer_science)), where different [operators](https://en.wikipedia.org/wiki/Operator_(computer_programming)) have different implementations depending on their arguments. Operator overloading is generally defined by a [programming language](https://en.wikipedia.org/wiki/Programming_language), a [programmer](https://en.wikipedia.org/wiki/Programmer), or both.



## Overriding 오버라이딩

부모 class의 method를 자식 class가 가져와서 자기만의 실행 방법을 만든다.



**Method overriding**, in [object-oriented programming](https://en.wikipedia.org/wiki/Object-oriented_programming), is a language feature that allows a [subclass](https://en.wikipedia.org/wiki/Subclass_(computer_science)) or child class to provide a specific implementation of a [method](https://en.wikipedia.org/wiki/Method_(computer_science)) that is already provided by one of its [superclasses](https://en.wikipedia.org/wiki/Superclass_(computer_science)) or parent classes.

