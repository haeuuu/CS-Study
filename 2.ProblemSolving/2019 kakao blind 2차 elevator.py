# p0
# AveWait: 7.833333, AveTravel: 8.000000, AveTotal: 15.833333, LastTs: 22, Status: OK

# p1
# AveWait: 32.880000, AveTravel: 38.290000, AveTotal: 71.170000, LastTs: 613, Status: OK

# p3
# AveWait: 16.850000, AveTravel: 25.660000, AveTotal: 42.510000, LastTs: 1902, Status: OK

import requests, time

url = 'http://localhost:8000'

def start(user, problem, count):
    uri = url + '/start' + '/' + user + '/' + str(problem) + '/' + str(count)
    return requests.post(uri).json()

def oncalls(token):
    uri = url + '/oncalls'
    return requests.get(uri, headers={'X-Auth-Token': token}).json()

def action(token, cmds):
    uri = url + '/action'
    return requests.post(uri, headers={'X-Auth-Token': token}, json={'commands': cmds}).json()

# enter 혹은 exit를 위한 dictionary 정의
# enter/exit를 위해서는 stop => open => enter/exit 명령을 순차적으로 해야함. 현재 status에 맞게 다음 action을 유도하기 위한 dictionary
command_enter = {'UPWARD': 'STOP', 'DOWNWARD': 'STOP', 'STOPPED': 'OPEN', 'OPENED': 'ENTER'}
command_exit = {'UPWARD': 'STOP', 'DOWNWARD': 'STOP', 'STOPPED': 'OPEN', 'OPENED': 'EXIT'}

def simulator():
    user, problem, count = 'tester', 2, 4
    limits = 8 # 최대 탑승 가능 인원
    lower, upper = 1, [5, 25, 25][problem] # 최저층, 최고층

    s = start(user, problem, count)
    token = s['token']
    print(token)

    # 이전 state를 저장하기 위한 list
    # elevator는 최저층 => 최고층 => 최저층 => 최고층 => ... 를 순차적으로 훑으면서 enter/exit를 하도록 설계되어있음.
    # 만약 enter/exit 후에 STOPPED 상태인 경우에는 이전에 움직이던 방향 그대로 다시 움직여야하므로 이를 list에 저장해놓은 후 필요할 때 활용
    prev_direction = ['UP', 'UP', 'UP', 'UP']

    while True:
        curr_calls = oncalls(token)
        if curr_calls['is_end']:
            break
        t = curr_calls['timestamp']

        # elevator를 순회하면서 상황에 맞게 base_commands를 수정한 후 한꺼번에 action에 보내자.
        base_commands = [{'elevator_id': 0, 'command': 'STOP'},
                         {'elevator_id': 1, 'command': 'STOP'},
                         {'elevator_id': 2, 'command': 'STOP'},
                         {'elevator_id': 3, 'command': 'STOP'}]

        # calls ; 이 변수를 쓴 이유가 약간 헷갈리실 것 같아서 예제로 설명해보겠습니다
        # 현재 모든 elevator가 3층에서 UPWARD이고, start가 3인 call이 3개인 경우 (id는 0,5,7이라고 하자.)
        # 만약 첫번째 elevator가 3개의 call을 모두 enter시킬 수 있는 상태라면 첫번째 elevator만 STOP하고 나머지는 그대로 계속 UP하는 것이 효율적.
        # 첫번째 elevator는 calls에 [0,5,7]을 append시키고 나머지 elevator에게 '3층에 있는 call은 더이상 고려하지 않아도 된다'를 알려주게 된다.
        # x in list 로 검사하는게 시간을 많이 잡아먹을거같아서 defaultdict 등으로 수정해야하나 했는데 엄청 느리진 않은거같아서 그냥 뒀습니다 !!!
        calls = []
        for elevator in curr_calls['elevators']:
            idx = elevator['id']

            # action에 보내기 위한 list. 만약 enter 또는 exit를 수행할 차례가 되어 call_ids가 채워지면 command에 추가한다.
            call_ids = []

            ########
            # EXIT #
            ########

            # 만약 탑승자가 한명이라도 있다면, 돌면서 내릴 수 있는지 확인한다.
            pass_flag = False # action이 결정되었는지 아닌지를 판단하기 위한 flag. 만약 내릴 사람이 한 명이라도 있는 경우에는 현재 action을 고정하고 다음 elevator로 넘어간다.
            for passenger in elevator['passengers']:
                if passenger['end'] == elevator['floor'] and passenger['id'] not in calls:  # 현재 층에서 내린다면 + 다른 elevator가 처리할 call이 아닌 경우
                    recommended_action = command_exit[elevator['status']]  # EXIT를 위해 거쳐야할 다음 action을 취한다.
                    base_commands[idx]['command'] = recommended_action

                    if recommended_action == 'EXIT':  # 만약 EXIT할 차례라면 call_ids를 채운다.
                        call_ids.append(passenger['id'])
                        calls.append(passenger['id'])
                    else:
                        pass_flag = True
                        break

            if pass_flag:
                continue

            if call_ids: # call_ids가 채워진 경우, 즉 exit하는 경우에 command dictionary에 추가해준다.
                base_commands[idx]['call_ids'] = call_ids
                continue

            #########
            # ENTER #
            #########

            # loop를 돌면서 처리할만한 call이 있는지 확인한다.
            for call in curr_calls['calls']:
                if call['start'] == elevator['floor']: # 같은 층에서 시작하는 요청이 있는 경우

                    # 만약 더 탑승할 수 없거나 이전 elevator가 처리한 call이면 continue
                    if len(elevator['passengers']) + len(call_ids) >= limits or call['id'] in calls:
                        continue

                    # 탑승할 수 있는 상태라면
                    recommended_action = command_enter[elevator['status']]  # ENTER를 위한 action을 취하고
                    base_commands[idx]['command'] = recommended_action
                    calls.append(call['id']) # 현재 elevator에서 처리할 call임을 명시해주고

                    if recommended_action == 'ENTER':  # 만약 ENTER 차례라면 call_ids를 채운다.
                        call_ids.append(call['id'])
                    else:
                        pass_flag = True
                        break

            if pass_flag:
                continue

            if call_ids:
                base_commands[idx]['call_ids'] = call_ids
                continue

            ##############################
            # exit도 enter도 하지 못한 경우 #
            ##############################

            # 1. 열려 있는 경우에는 이동 or 대기를 위해 일단 닫는다
            if elevator['status'] == 'OPENED':
                base_commands[idx]['command'] = 'CLOSE'
                continue

            # 나머지 경우에는 기존 방향을 유지하거나 최고층/최저층에서 전환한다.
            if elevator['floor'] == upper or elevator['floor'] == lower: # 최고층이나 최저층인 경우
                if elevator['status'] == 'STOPPED': # 멈춰있다면, 아래 혹은 위로 이동한다.
                    base_commands[idx]['command'] = ['UP', 'DOWN'][elevator['floor'] == upper]
                    prev_direction[idx] = base_commands[idx]['command'] # 이전 이동 방향이 바뀌었으므로 그에 맞게 수정해준다.
                else:
                    base_commands[idx]['command'] = 'STOP' # 멈춰있지 않다면 방향 전환을 위해 멈춘다.
            else:
                base_commands[idx]['command'] = prev_direction[idx] # 모두 아니라면 이전 방향을 유지한채 이동한다.

        action(token, base_commands)

if __name__ == '__main__':
    simulator()
