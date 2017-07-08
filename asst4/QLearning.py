# Author: erlie.shang@uwaterloo.ca
# T[s,s',a] = Pr(s'|s,a)
from gridWorld import gridWorld
import random
FACTOR = 0.99
actions = ['up', 'down', 'left', 'right']


def select_action(actions):
    ret = None
    Q = float('-inf')
    for i in range(4):
        if actions[i] > Q:
            ret = i
            Q = actions[i]
    return ret


def next_state(state, action, t, g):
    p = [0.0 for i in range(17)]
    for i in range(17):
        p[i] = t[state, i, action]
    x = random.random()
    for i in range(17):
        x -= p[i]
        if x <= 0:
            if random.random() > g:
                return i
            else:
                return random.randint(0, 3)
    return None


def q_learning(greedy):
    t, r = gridWorld(0.9, 0.05)
    q_table = [[0.0, 0.0, 0.0, 0.0] for i in range(17)]
    q_counter = [[0.0, 0.0, 0.0, 0.0] for i in range(17)]
    for i in range(10000):
        state = 4
        while state != 16:
            action = select_action(q_table[state])
            ns = next_state(state, action, t, greedy)
            q_counter[state][action] += 1
            q_table[state][action] += ((1 / q_counter[state][action]) * (
                r[state] + FACTOR * q_table[ns][select_action(q_table[ns])] - q_table[state][action]))
            state = ns
    return q_table


def main():
    q1 = q_learning(0.05)
    print "when greedy = 0.05"
    for i in range(17):
        print "state", i, "action:", actions[select_action(q1[i])], "Q value:", q1[i][select_action(q1[i])]
    q2 = q_learning(0.2)
    print "when greedy = 0.2"
    for i in range(17):
        print "state", i, "action:", actions[select_action(q2[i])], "Q value:", q2[i][select_action(q2[i])]


if __name__ == "__main__":
    main()
