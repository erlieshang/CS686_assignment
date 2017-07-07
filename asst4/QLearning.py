# Author: erlie.shang@uwaterloo.ca
# T[s,s',a] = Pr(s'|s,a)
from gridWorld import gridWorld
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

def main():
    t, r = gridWorld(0.9, 0.05)
    a = 0.5
    q_table = [[0, 0, 0, 0] for i in range(17)]
    # start from state 4
    action = select_action(q_table[4])
    q_table[4][action] += a*(r[4] + FACTOR*max() - q_table[4][action])


if __name__ == "__main__":
    main()
