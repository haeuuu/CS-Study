## Git을 사용하며 만난 이슈를 정리합니다.

### 깃 충돌 해결하기
* remote와 local에서 같은 파일에 대한 commit을 생성했을 때 발생한다.
1. `git pull origin master` : remote의 변경 사항을 가져온다.  
    * 이 때 local에서 다른 파일을 수정 중이고, 아직 commit하지 않았다면 다 commit 해주자.
```
  Auto-merging TitleBasedPlyGenerator.py
CONFLICT (content): Merge conflict in TitleBasedPlyGenerator.py
Automatic merge failed; fix conflicts and then commit the result.
```
2. `git status`에서 `Unmerged paths`를 확인한다.
    * 충돌이 일어난 파일 list를 확인할 수 있다.
    * 자동으로 해결 가능하지 않은 경우 생성된다.
```
On branch master
Your branch and 'origin/master' have diverged,
and have 14 and 1 different commits each, respectively.
  (use "git pull" to merge the remote branch into yours)
You have unmerged paths.
  (fix conflicts and run "git commit")

Unmerged paths:
  (use "git add <file>..." to mark resolution)

        both modified:   TitleBasedPlyGenerator.py

Untracked files:
  (use "git add <file>..." to include in what will be committed)

        .gitignore

no changes added to commit (use "git add" and/or "git commit -a")

```
3. Unmerged인 파일을 열어보자.
* `<<<<<<<<`와 `>>>>>>>>` 로 쌓여진 부분은 수정을,
* `========`로 표시된 부분은 삭제를 하면 된다.

4. 수정 사항을 `add`, `commit`하자.
5. `git push origin master`


### 지워버린 파일 복구하기
```
git checkout filename
```

### 직전 commit 되돌리기
```
git reset HEAD~1
```
직전 commit 1개를 취소할 수 있다.

### push 되돌리기
github 페이지에서 history를 열어보면 commit id가 있다. 복사버튼으로 복사해서
```
git reset commit_id
```

### repo의 commit list 또는 특정 파일의 commit list 확인하기
```
git log
git log filename
```

### 직전 commit과 코드 비교하기
```
git diff filename
```
enter를 통해 다음 코드를 확인하고 q로 빠져나온다.
