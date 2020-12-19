# :christmas_tree: @property 알아보기



## [private] class 내부에서만 연산 가능한 변수

언더바 두개를 통해 private한 변수를 만들 수 있다.

외부에서 접근할 수 없다.

```python
class MyClass:
    def __init__(self):
	    self.__secret_data = 'private data'
    
a = MyClass()
a.__data # AttributeError : 'MyClass' obj has no attr '__data'`
```



## getter, setter로 private 접근/수정하기

이 변수를 접근/수정하기 위해서는 class 내부에 `getter`와 `setter` method가 정의되어야 한다.

```python
class MyClass:
    def __init__(self):
	    self.__secret_data = 'private data'
    
    def get_data(self):
        """private 변수인 __secret_data 접근한다."""
        return self.__secret_data
    
    def set_data(self,new_data):
        """private 변수인 __secret_data 수정한다."""
        self.__secret_data = new_data
    
    
a = MyClass()
a.get_data() # return 'private data'
```



#### 왜 필요한가?

OOP의 4대 원칙 중 하나인 캡슐화와 관련이 있다.

class 내부 변수는 보호되어야 한다. 외부에서 마음대로 접근/수정이 가능하다면 로직에 오류가 생긴다.

public/private한 method와 attr를 구분할 필요가 있다.



## @property, @data.setter

python에서는 decorater를 통해 `getter, setter`를 구현할 수도 있다.

각각 `@property`와 `@[property가 달린 method 이름].setter`에 해당한다.

```python
class MyClass:
    def __init__(self):
	    self.__secret_data = 'private data'
    
    @property
    def secret_data(self):
        """private 변수인 __secret_data 접근한다."""
        return self.__secret_data
    
    @secret_data.setter
    def secret_data(self,new_data):
        """private 변수인 __secret_data 수정한다."""
        self.__secret_data = new_data
    
    
a = MyClass()
a.secret_data # return 'private data'
```

이 때 `@secret_data.setter`를 선언하기 전에, `secret_data` 를 property로 만들어주어야 한다.

(즉 변수를 수정하기 위해서는 가져올 수 있어야 한다. 가져오기 기능만 구현하는 것은 가능)



#### @property를 붙여주면 method를 attr처럼 사용할 수 있게 된다.



# :deciduous_tree: 그래서 언제 유용한데?

만약 어떤 변수가 값에 제약이 있다고 하자.

사용자가 애초에 그런 값을 입력하지 않으면 좋겠지만, 불가능한 값이 들어왔을 때 내부 변수가 수정되지 않도록 보호해주고 싶을 수 있다. (값이 0이나 음수가 되면 division by zero 등 에러가 발생한다던가 .. 등의 이유로?)



`class Person`의 `age`에 음수가 들어오면 안될 때, 다음처럼 코딩할 수 있다.

```python
class Person:
    def __init__(self):
        self._age = None
        
    @property
    def age(self):
        return self._age
    
    @age.setter
    def age(self, new_age):
        
        # 음수가 선언되면 내부 변수를 변경하지 않고 종료한다.
        if new_age < 0:
            print('[INFO] Not Avaliable Value.')
            return
        
        self._age = new_age
        print('[INFO] Changed.')
```

```python
haeu = Person()
print(haeu.age)

haeu.age = 25
print(haeu.age) # 25

haeu.age = -20 # 불가능한 입력. 자동으로 튕기게 만들 수 있다.
print(haeu.age) # 25
```

```
실행 결과
	None

    [INFO] Changed.
    25
    
    [INFO] Not Avaliable Value.
    25
```



혹은 딥러닝 모델을 학습했을 때 유용하게 쓰일거같다!

MF 등을 학습해서 얻은 weight(user factor, item factor)들은 함부로 수정되어서는 안된다. 접근만이 가능하도록 `@property`를 통해 보호할 필요가 있다.implicit 패키지의 `matrix_factorization_base.py`에도 학습 후 생성된 값을 보호하기 위해 `@property`를 사용하고 있다.



# :palm_tree: 주의할 점

파이썬에서는 잘못된 접근에는 에러를 띄워주지만, 잘못된 선언에 대해서는 에러를 띄우지 않는다. 즉

**1 ) 접근이 불가능한 경우** => ERROR !

```python
class MyClass:
    def __init__(self):
	    self.__secret_data = 'private data'
    
a = MyClass()
a.__data # AttributeError : 'MyClass' obj has no attr '__data'`
```

**2 ) getter은 가능하지만 setter은 정의되지 않은 경우**

잘못 선언하더라도 에러가 나지 않는, 즉 변경된 척 하지만 사실은 그렇지 않다. 조심해야한다!

```python
class MyClass:
    def __init__(self):
	    self.__secret_data = 'private data'
    
    @property
    def secret_data(self):
        """private 변수인 __secret_data 접근한다."""
        return self.__secret_data
    
a = MyClass()
a.secret_data # return 'private data'

a.__secret_data = '무단 침입 !!!'
# 에러가 나지 않는다 !!
# 그러나 변수가 변경된 것은 아니다,
a.secret_data # 여전히 'private data' return
```

그러나 아래와 같은 명령은 error가 발생한다.

```python
a.secret_data = '무단 침입 !!!'
# AttributeError: can't set attribute
# 위 명령은 [method 이름].setter를 호출한다. 정의되지 않았으므로 set이 불가능해서 ERROR
```



#### Reference

https://nowonbun.tistory.com/660

https://hamait.tistory.com/827
