# Docker

컨테이너를 관리하는! 오픈소스 가상화! 플랫폼.



## Introduction

대부분의 software는 운영체제를 기반으로 만들어진다.

software의 실행을 위해서는 os, library, 실행을 위한 파일 등으로 구성된 **실행 환경이 필요**하다.

그런데 하나의 시스템 위에서 **둘 이상의 software를 동시에 실행하려고 한다면 문제가 발생**할 수 있다.

ex : 동일한 library를 사용하지만 다른 버전이 필요한 경우.

ex2 : 두 software의 운영 체제가 다를 경우

⇒ 이를 간단히 해결하는 방법은? : 각각의 software를 위한 시스템을 따로 준비하는 것!

⇒ 잠깐! 그럼 software가 100개면, 100개의 시스템을 준비해야 한다! 비효율적!!!

⇒ 컨테이너 개념을 도입해서 이 문제를 해결하자.



### 📦 Container 컨테이너

개별 software의 실행에 **필요한 환경을 독립적으로 운용**할 수 있도록 하는 가상화 기술.

- 프로그램과 실행 환경을 컨테이너로 추상화한다.
- 동일한 인터페이스를 제공할 수 있으므로 **프로그램의 배포 및 관리가 단순**해진다.
  - 규격화된 컨테이너에 **원하는 물품을 담아 어디든 싣고 옮길 수 있는 것처럼**, 다양한 프로그램/실행환경을 컨테이너라는 개념으로 추상화하여 프로그램의 배포/관리를 단순하게 해준다.
- 백엔드, 데이터 베이스 서버, 메시지 큐 등 어떤 프로그램도 컨테이너로 추상화할 수 있다.
- 조립PC,  AWS, Azure, google cloud등 어디서든 실행할 수 있다.
- 하나의 서버에서 여러개의 컨테이너를 실행할 수 있다.



## 기존의 가상화 방식

**기존 : OS 전체 혹은 일부를 가상화 했다.**

VirtualBox와 같은 가상 머신은, Host OS위에 Guest OS 전체를 가상화한다.

아래 그림을 보면 각각의 `App`을 위한 환경이 따로 구축되어 있는 것을 확인할 수 있다.

무겁고 느리다.



**Docker : 프로세스를 격리한다.**

![Docker%20eb7aea2834d643d2a1a39eeb42156859/Untitled.png](Docker%20eb7aea2834d643d2a1a39eeb42156859/Untitled.png)



## 🖼️ Image

- 컨테이너 실행을 위한 파일, 설정값 등을 포함하고 있는 것.
- immutable하다.
- 컨테이너는 이미지를 실행한 상태라고 볼 수 있다.
  - 추가되거나 변하는 값은 컨테이너에 저장된다.
  - 하나의 이미지를 이용해서 여러개의 컨테이너를 생성할 수 있다.

왜 굳이 Image라고 할까 궁금했는데, **필요한 설정들을 '캡쳐'해놓는다**는 설명을 보니 이해가 좀 되는듯!!

어쨌든, 이처럼 이미지는 **컨테이너를 실행하기 위한 모든 파일,설정**을 가지고 있으므로! <의존성을 체크하고 ... 없는 것은 설치하고 ... > 등의 과정을 거칠 필요가 없다.

그저 이미지를 다운받아서 컨테이너를 생성하면 된다!



## 🗃️ Layer

이미지는 모든 정보를 저장해야하므로 용량이 수백MB이다.

만약 이미 다운 받아놓은 이미지에 파일이 하나 추가되어서, 새로운 이미지를 받아야 한다면?

파일 하나를 위해 수백메가짜리 이미지를 **다시 다운 받는 것은 비효율적**이다.

⇒ Layer 개념을 도입하자!

- 이미지는 여러개의 **읽기 전용 레이어 read-only** 로 구성된다.
- 파일이 하나 추가되면 새로운 레이어가 생성된다.
- 만약 A,B,C,D layer 중에서 B에 대한 내용이 수정되었다면, **B만 다시 다운받으면 된다!**



**컨테이너를 생성할 때는?**

- 컨테이너가 실행되면서 **생성/변경되는 내용을 저장**할 수 있어야 한다!
- 기존의 layer 위에 **읽기-쓰기 레이어 read-write를 추가**한다.
- **이미지 레이어는 그대로 사용**하면서, **수정된 내용은 읽기-쓰기 레이어에 저장**되므로, 여러개의 컨테이너를 생성해도 **최소한의 용량만 사용**한다.
  - 🤔 하나의 레이어마다 갖는건가? 아니면 전체 이미지에 대한 읽기-쓰기 레이어 하나만 생성되는건가?



**이미지 관리하기**

- url 방식으로 관리한다. tag를 붙일 수 있다.

![Docker%20eb7aea2834d643d2a1a39eeb42156859/Untitled%201.png](Docker%20eb7aea2834d643d2a1a39eeb42156859/Untitled%201.png)



### 🐋 **Dockerfile**

- 이미지를 만들 때,
  `Dockerfile` 이라는 파일에,
  `DSL(Domain-specific language)`이라는 언어를 이용해서,
  이미지 생성 과정을 적는다.
- `requirements.txt` 의 진화 버전? 이라는 생각이 든다! 패키지 설치 등의 설정 사항을 `Dockerfile`로 관리한다. 누구나 이미지 생성 과정을 보고 수정할 수 있다.



### 🌿 **Docker Hub**

- 큰 용량의 도커 이미지를 서버에 저장하고 관리하는 것은 쉽지 않다.
- Docker Hub에서는 이를 무료로 할 수 있게 도와준다!



## 그외 메모

- Host OS - 가상머신 - 도커로 연결되어있지만, 사용자는 가상머신을 거치지 않는 것처럼 사용할 수 있다.

  - 디렉토리 연결이나 포트 연결 시에 OS-가상머신 연결 + 가상머신 - 도커 연결을 해야하지만, 이러한 부분을 자연스럽게 챡챡 처리해준다.

- Docker는 Client, Server 역할을 모두 할 수 있다.

  ![Docker%20eb7aea2834d643d2a1a39eeb42156859/Untitled%202.png](Docker%20eb7aea2834d643d2a1a39eeb42156859/Untitled%202.png)

  1. `$ docker run xxx` 등으로 Docker에게 명령하면 (client)
  2. docker server로 명령을 전송하고
  3. 이 결과를 client가 받아 결과를 terminal에 보여준다.



# Docker 기본 명령어

## 📦 Container 관련 명령어 - 1

**`run` 컨테이너 생성하기**

```docker
$ docker run [Options] Image[:Tag|@Digest] [Command] [ARG ...]
```

- `-d` : **detached mode**. background 모드
- `-p` : **port forwarding.**호스트와 컨테이너의 **포트를 연결**.
- `-v` :  **(volume의 v인가?) mount**. 호스트와 컨테이너의 **디렉토리를 연결**.
- `-e` : **environment**. 컨테이너 내에서 사용할 환경 변수 설정.
- `-name` : 컨테이너 이름 설정
- `-rm` : 프로세스 종료시 컨테이너 자동 **제거**
- `-it` = `-i` + `-t` : 터미널 입력을 위한 옵션
- `-link` : 컨테이너 연결. `[컨테이너이름:별칭]`



`ps` **컨테이너 목록 확인하기. process?**

```docker
$ docker ps [OPTIONS]
```

- `docker ps` : **실행중**인 컨테이너 목록을 띄운다.
- `docker ps -a` : 종료된 컨테이너 (`Exited (0)` ) 까지 띄워준다.
  - 종료 상태에도 읽기-쓰기 레이어는 살아있다.
  - 종료 상태의 컨테이너도 언제든 다시 시작할 수 있다.
  - 컨테이너를 삭제한다면 이 목록에서 제거된다.



**`stop` 컨테이너 중지하기**

```docker
$ docker stop [Options] Container [Container...]

$ docker stop ${CONTAINER_ID} ${CONTAINER_ID_2} # ${}까지 하라는건가,,,? 헤
```

- 여러개의 컨테이너를 한번에 중지할 수 있다.

- 컨테이너의 ID 또는 이름을 입력하면 된다. 위의 `ps` 명령어를 이용해서 확인하면 된다.

  - 만약 전부 입력하지 않아도 구분 가능하다면, 앞 몇글자만 입력해도 된다.

    ID는 총 64자리.  `bdfsac...`에서 `bdf` 만 써서 지울 수 있다.



**`rm` 컨테이너 제거하기. remove**

```docker
$ docker rm [Options] Container [Container ...]
```

- 여러개의 컨테이너를 한꺼번에 삭제할 수 있다.

- 중지된 컨테이너를 한꺼번에 삭제하고 싶다면?

  `docker ps -a -q -f status=exited` 를 이용하면, 중지된 container의 id를 한번에 가져올 수 있다.

  - `-q`
  - `-f`

  ```docker
  $ docker rm -v $(docker ps -a -q -f status=exited)
  ```



## 🖼️ Image 관련 명령어

**`images` 이미지 목록 확인하기**

```docker
$ docker images [Options] [Repository[:Tag]]
```

```docker
$ docker images
```



**`pull` 이미지 다운로드**

```docker
$ docker pull [Options] Name[:Tag|@Digest]
```

- 이미지를 최신 버전으로 다시 다운 받고 싶을 때 다시 `pull` 하면 된다.
  - 아예 처음부터 다시 받는다는건가? 아니면 update된 layer만 다시 받는건가

```docker
$ docker pull ubuntu:14.04
```



**`rmi` 이미지 삭제하기. remove images**

```docker
$ docker rmi [Options] Image [Image ... ]
```

- 해당 이미지로 생성된 컨테이너가 실행중이라면, 삭제되지 않는다.
  - 왜냐!! 컨테이너는 이미지의 레이어를 기반으로 실행중이니까.



### Example1 : Ubuntu 16.04 이미지를 다운받고, 컨테이너를 생성해보자.

**1 ) image run 해보기**

```docker
$ docker run ubuntu:16.04
```

`docker run Image` 형태로 명령했다. 다음 과정을 거친다.

1. 명령한 이미지 `ubuntu:16.04` 가 local에 있는지 확인한다.
2. 없다면 다운로드 `Pull` 한다.
3. 컨테이너를 생성 `Create` 한다.
4. 시작 `Start` 한다.

그러나, 그저 "이미지를 실행 시켜. 컨테이너를 생성해."라는 명령만 했다.

그러므로 이미지 다운로드가 끝나고, 컨테이너가 **생성되자마자 종료**된다!
(삭제되었다고는 안했다!! 그냥 꺼진 것.)

**컨테이너는 프로세스이므로, 실행중인 프로세스가 없으면 컨테이너는 종료된다.**



**2 ) `/bin/bash` 명령어 추가해보기**

```docker
$ docker run [--rm -it] [ubuntu:16.04] [/bin/bash]
```

1. 컨테이너 내부에서 bash shell을 실행할 것이다.
2. 터미널에 입력해보고 싶으므로 `-it` 옵션도 추가한다.
3. 프로세스가 종료되면 컨테이너를 자동 삭제시키기 위해 `--rm` 도 추가한다.

```docker
root@....:/# cat /etc/issue
>> Ubuntu 16.04.1 LTS \n \1

root@....:/# ls
>> bin dev hom lib64 mnt proc run srv tmp var

root@....:/# exit
```

위처럼 입력해보자. Ubuntu임을 확인할 수 있다.

exit로 bash shell을 종료하면 컨테이너도 종료되고 자동으로 삭제된다.



## 📦 Container 관련 명령어 - 2

**`logs` 컨테이너 로그 보기**

```docker
$ docker logs [OPT] Container
```

컨테이너가 정상적으로 동작하는지 확인하려면 log를 체크해보자.

- 기본적으로는 모든 로그를 전부 print한다.
- `-tail k`  : 마지막 k줄만 출력하라.
- `-f` : 실시간으로 생성되는 로그를 확인하자. `ctrl+c` 로 로그 보기를 중지할 수 있다.



## Reference

- [https://subicura.com/2017/01/19/docker-guide-for-beginners-1.html](https://subicura.com/2017/01/19/docker-guide-for-beginners-1.html)
- [https://subicura.com/2017/01/19/docker-guide-for-beginners-2.html](https://subicura.com/2017/01/19/docker-guide-for-beginners-2.html)