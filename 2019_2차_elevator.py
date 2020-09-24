# calls를 list로 처리한 경우

# p0
# AveWait: 7.833333, AveTravel: 8.000000, AveTotal: 15.833333, LastTs: 22, Status: OK

# p1
# AveWait: 32.880000, AveTravel: 38.290000, AveTotal: 71.170000, LastTs: 613, Status: OK

# p3

import requests, time
from collections import defaultdict

url = 'http://localhost:8000'
url = 'http://2019-kakao-practice.encrypted.gg:8000'

def start(user, problem, count):
    uri = url + '/start' + '/' + user + '/' + str(problem) + '/' + str(count)
    return requests.post(uri).json()


def oncalls(token):
    uri = url + '/oncalls'
    return requests.get(uri, headers={'X-Auth-Token': token}).json()


def action(token, cmds):
    uri = url + '/action'
    return requests.post(uri, headers={'X-Auth-Token': token}, json={'commands': cmds}).json()


command_enter = {'UPWARD': 'STOP', 'DOWNWARD': 'STOP', 'STOPPED': 'OPEN', 'OPENED': 'ENTER'}
command_exit = {'UPWARD': 'STOP', 'DOWNWARD': 'STOP', 'STOPPED': 'OPEN', 'OPENED': 'EXIT'}

user, problem, count = 'tester', 0, 4
limits = 8
lower, upper = 1, [5, 25, 25][problem]

s = start(user, problem, count)
token = s['token']
print(token)

prev_direction = ['UP', 'UP', 'UP', 'UP']

while True:
    curr_calls = oncalls(token)
    if curr_calls['is_end']:
        break
    t = curr_calls['timestamp']
    print(t,'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    print(curr_calls['calls'][-1:])
    base_commands = [{'elevator_id': 0, 'command': 'STOP'},
                     {'elevator_id': 1, 'command': 'STOP'},
                     {'elevator_id': 2, 'command': 'STOP'},
                     {'elevator_id': 3, 'command': 'STOP'}]

    calls = []
    for elevator in curr_calls['elevators']:
        idx = elevator['id']
        call_ids = []

        ########
        # EXIT #
        ########

        # 만약 탑승자가 한명이라도 있다면, 돌면서 내릴 수 있는지 확인한다.
        pass_flag = False
        for passenger in elevator['passengers']:
            if passenger['end'] == elevator['floor'] and passenger['id'] not in calls:  # 현재 층에서 내린다면
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

        if call_ids:
            base_commands[idx]['call_ids'] = call_ids
            continue

        ##########
        # ENTER #
        #########

        for call in curr_calls['calls']:
            if call['start'] == elevator['floor']:
                if len(elevator['passengers']) + len(call_ids) >= limits or call['id'] in calls:
                    continue
                # 탑승할 수 있는 상태라면
                recommended_action = command_enter[elevator['status']]  # ENTER를 위한 action을 취하고
                base_commands[idx]['command'] = recommended_action
                calls.append(call['id'])

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

        #################################
        # exit도 enter도 하지 못한 경우 #
        #################################

        # 1. 열려 있는 경우에는 이동 or 대기를 위해 일단 닫는다
        if elevator['status'] == 'OPENED':
            base_commands[idx]['command'] = 'CLOSE'
            continue

        # 나머지 경우에는 기존 방향을 유지하거나 최고층/최저층에서 전환한다.
        if elevator['floor'] == upper or elevator['floor'] == lower:
            if elevator['status'] == 'STOPPED':
                base_commands[idx]['command'] = ['UP', 'DOWN'][elevator['floor'] == upper]
                prev_direction[idx] = base_commands[idx]['command']
            else:
                base_commands[idx]['command'] = 'STOP'
        else:
            base_commands[idx]['command'] = prev_direction[idx]

    action(token, base_commands)
