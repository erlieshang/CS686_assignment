# Author: erlie.shang@uwaterloo.ca
# T[s,s',a] = Pr(s'|s,a)
from gridWorld import gridWorld


class Node(object):
    def __init__(self):
        self.value = 0
        self.action = None

FACTOR = 0.99
actions = ['up', 'down', 'left', 'right']


def get_policy(world):
    t, r = world
    table = list()
    horizon = [Node() for i in range(17)]
    for i in range(17):
        horizon[i].value = r[i]
    table.append(horizon)
    while True:
        row = [Node() for i in range(17)]
        for i in range(17):
            value = float('-inf')
            action = None
            for a in range(4):
                tmp = 0
                for j in range(17):
                    tmp += t[i, j, a] * table[-1][j].value
                if tmp > value:
                    value = tmp
                    action = a
            row[i].value = value * FACTOR + r[i]
            row[i].action = action
        table.append(row)
        done = True
        for i in range(17):
            if abs(table[-1][i].value - table[-2][i].value) >= 0.01:
                done = False
                break
        if done:
            break
    return table[-1]


def main():
    policy1 = get_policy(gridWorld(0.9, 0.05))
    policy2 = get_policy(gridWorld(0.8, 0.1))
    print "For a = 0.9 and b = 0.05"
    print "optimal policies:"
    for i in range(17):
        print 'state', i, 'value:', policy1[i].value, 'action:', actions[policy1[i].action]
    print "For a = 0.8 and b = 0.1"
    print "optimal policies:"
    for i in range(17):
        print 'state', i, 'value:', policy2[i].value, 'action:', actions[policy2[i].action]


if __name__ == "__main__":
    main()
