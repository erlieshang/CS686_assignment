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


def next_state(state, action, t):
    p = [0.0 for i in range(17)]
    for i in range(17):
        p[i] = t[state, i, action]
    x = random.random()
    for i in range(17):
        x -= p[i]
        if x <= 0:
            return i
    return None


def main():
    t, r = gridWorld(0.9, 0.05)
    a = 0.5
    q_table = [[0, 0, 0, 0] for i in range(17)]
    # start from state 4
    action = select_action(q_table[4])
    ns = next_state(4, action, t)
    q_table[4][action] += a * (r[4] + FACTOR * q_table[ns][select_action(q_table[ns])] - q_table[4][action])


if __name__ == "__main__":
    main()
