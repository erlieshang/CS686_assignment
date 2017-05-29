#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" a Sudoku Solver """
import time
import random
import copy

__author__ = 'erlie.shang@uwaterloo.ca'


class Stats:
    def __init__(self, sequence):
        # sequence of numbers we will process
        # convert all items to floats for numerical processing
        self.sequence = [float(item) for item in sequence]

    def sum(self):
        if len(self.sequence) < 1:
            return None
        else:
            return sum(self.sequence)

    def count(self):
        return len(self.sequence)

    def min(self):
        if len(self.sequence) < 1:
            return None
        else:
            return min(self.sequence)

    def max(self):
        if len(self.sequence) < 1:
            return None
        else:
            return max(self.sequence)

    def avg(self):
        if len(self.sequence) < 1:
            return None
        else:
            return sum(self.sequence) / len(self.sequence)

    def median(self):
        if len(self.sequence) < 1:
            return None
        else:
            self.sequence.sort()
            return self.sequence[len(self.sequence) // 2]

    def stdev(self):
        if len(self.sequence) < 1:
            return None
        else:
            avg = self.avg()
            sdsq = sum([(i - avg) ** 2 for i in self.sequence])
            stdev = (sdsq / (len(self.sequence) - 1)) ** .5
            return stdev

    def percentile(self, percentile):
        if len(self.sequence) < 1:
            value = None
        elif (percentile >= 100):
            sys.stderr.write('ERROR: percentile must be < 100.  you supplied: %s\n' % percentile)
            value = None
        else:
            element_idx = int(len(self.sequence) * (percentile / 100.0))
            self.sequence.sort()
            value = self.sequence[element_idx]
        return value


class SudokuSolverB(object):

    def __init__(self, puzzle):
        self._result = copy.deepcopy(puzzle)
        self._solved = False
        self._time_cost = 0
        self._node = 0

    def reinit(self, puzzle):
        self._result = copy.deepcopy(puzzle)
        self._solved = False
        self._time_cost = 0
        self._node = 0

    def is_complete(self, asst):
        for i in asst:
            for j in i:
                if j == 0:
                    return False
        return True

    def check(self, asst, x, y, value):
        for i in range(9):
            if asst[x][i] == value:
                return False
            if asst[i][y] == value:
                return False
        x_start = x / 3 * 3
        y_start = y / 3 * 3
        for i in range(x_start, x_start + 3):
            for j in range(y_start, y_start + 3):
                if asst[i][j] == value:
                    return False
        return True

    def find_unassigned(self, asst):
        for x in range(9):
            for y in range(9):
                if asst[x][y] == 0:
                    return [x, y]

    def backtracking(self, asst):
        if self.is_complete(asst):
            return asst
        x, y = self.find_unassigned(asst)
        values = range(1, 10)
        random.shuffle(values)
        for value in values:
            if self.check(asst, x, y, value):
                asst[x][y] = value
                self._node += 1
                result = self.backtracking(asst)
                if result:
                    return result
                asst[x][y] = 0
        return False

    def solve(self):
        start = time.clock()
        self._node = 0
        if len(self._result) != 9:
            print 'invalid size'
            return
        for i in self._result:
            if len(i) != 9:
                print 'invalid size'
                return
        self._result = self.backtracking(self._result)
        self._solved = True
        self._time_cost = (time.clock() - start) * 1000

    def get_answer(self):
        if self._solved:
            return self._result
        else:
            print 'Run solve first!'

    def get_time_cost(self):
        return self._time_cost

    def get_node(self):
        return self._node

    def print_answer(self):
        if self._solved:
            for i in self._result:
                for j in i:
                    print repr(j).rjust(2),
                print
            print 'time cost is', self._time_cost, 'ms and node # is', self._node
        else:
            print 'Run solve first!'


class SudokuSolverBFC(SudokuSolverB):

    def __init__(self, puzzle):
        SudokuSolverB.__init__(self, puzzle)
        self._table = [list() for i in range(9)]
        for i in range(9):
            self._table[i] = [set(range(1, 10)) for j in range(9)]
        for x in range(9):
            for y in range(9):
                if self._result[x][y] != 0:
                    self.update_table(x, y, self._result[x][y], self._table)

    def reinit(self, puzzle):
        SudokuSolverB.reinit(self, puzzle)
        self._table = [list() for i in range(9)]
        for i in range(9):
            self._table[i] = [set(range(1, 10)) for j in range(9)]
        for x in range(9):
            for y in range(9):
                if self._result[x][y] != 0:
                    self.update_table(x, y, self._result[x][y], self._table)

    def update_table(self, x, y, value, table):
        ret = list()
        ret.append(table[x][y])
        table[x][y] = set([value])

        for i in range(9):
            if i != y and value in table[x][i]:
                table[x][i].remove(value)
                ret.append((x, i))
            if i != x and value in table[i][y]:
                table[i][y].remove(value)
                ret.append((i, y))

        x_start = x / 3 * 3
        y_start = y / 3 * 3
        for i in range(x_start, x_start + 3):
            for j in range(y_start, y_start + 3):
                if i != x and j != y and value in table[i][j]:
                    table[i][j].remove(value)
                    ret.append((i, j))
        return ret

    def rollback_table(self, x, y, value, table, original):
        table[x][y] = original[0]

        for axis in original[1:]:
            table[axis[0]][axis[1]].add(value)

    def has_conflict(self, table):
        for x in table:
            for y in x:
                if len(y) == 0:
                    return True
        return False

    def backtracking(self, asst, table):
        if self.is_complete(asst):
            return asst
        if self.has_conflict(table):
            return False
        x, y = self.find_unassigned(asst)
        values = range(1, 10)
        random.shuffle(values)
        for value in values:
            if value in table[x][y]:
                asst[x][y] = value
                original_info = self.update_table(x, y, value, table)
                self._node += 1
                result = self.backtracking(asst, table)
                if result:
                    return result
                asst[x][y] = 0
                self.rollback_table(x, y, value, table, original_info)
        return False

    def solve(self):
        start = time.clock()
        self._node = 0
        if len(self._result) != 9:
            print 'invalid size'
            return
        for i in self._result:
            if len(i) != 9:
                print 'invalid size'
                return
        self._result = self.backtracking(self._result, self._table)
        self._solved = True
        self._time_cost = (time.clock() - start) * 1000


class SudokuSolverBFCH(SudokuSolverBFC):

    def find_unassigned(self, asst, table):
        min = 9
        ret = [0, 0]
        for x in range(9):
            for y in range(9):
                if asst[x][y] == 0 and len(table[x][y]) < min:
                    ret = [x, y]
                    min = len(table[x][y])
        return ret

    def order_value(self, x, y, table):
        order = list()
        for value in table[x][y]:
            original_info = self.update_table(x, y, value, table)
            cnt = 0
            for i in table:
                for j in i:
                    cnt += len(j)
            order.append((value, cnt))
            self.rollback_table(x, y, value, table, original_info)
        sorted(order, lambda a, b: -1 if a[1] > b[1] else 1)
        ret = list()
        for v in order:
            ret.append(v[0])
        return ret

    def backtracking(self, asst, table):
        if self.is_complete(asst):
            return asst
        if self.has_conflict(table):
            return False
        x, y = self.find_unassigned(asst, table)
        for value in self.order_value(x, y, table):
            asst[x][y] = value
            original_info = self.update_table(x, y, value, table)
            self._node += 1
            result = self.backtracking(asst, table)
            if result:
                return result
            asst[x][y] = 0
            self.rollback_table(x, y, value, table, original_info)
        return False

if __name__ == '__main__':
    puzzle = [
        [[0, 6, 1, 0, 0, 0, 0, 5, 2],
         [8, 0, 0, 0, 0, 0, 0, 0, 1],
         [7, 0, 0, 5, 0, 0, 4, 0, 0],
         [9, 0, 3, 6, 0, 2, 0, 4, 7],
         [0, 0, 6, 7, 0, 1, 5, 0, 0],
         [5, 7, 0, 9, 0, 3, 2, 0, 6],
         [0, 0, 4, 0, 0, 9, 0, 0, 5],
         [1, 0, 0, 0, 0, 0, 0, 0, 8],
         [6, 2, 0, 0, 0, 0, 9, 3, 0]],

        [[5, 0, 0, 6, 1, 0, 0, 0, 0],
         [0, 2, 0, 4, 5, 7, 8, 0, 0],
         [1, 0, 0, 0, 0, 0, 5, 0, 3],
         [0, 0, 0, 0, 2, 1, 0, 0, 0],
         [4, 0, 0, 0, 0, 0, 0, 0, 6],
         [0, 0, 0, 3, 6, 0, 0, 0, 0],
         [9, 0, 3, 0, 0, 0, 0, 0, 2],
         [0, 0, 6, 7, 3, 9, 0, 8, 0],
         [0, 0, 0, 0, 8, 6, 0, 0, 5]],

        [[0, 4, 0, 0, 2, 5, 9, 0, 0],
         [0, 0, 0, 0, 3, 9, 0, 4, 0],
         [0, 0, 0, 0, 0, 0, 0, 6, 1],
         [0, 1, 7, 0, 0, 0, 0, 0, 0],
         [6, 0, 0, 7, 5, 4, 0, 0, 9],
         [0, 0, 0, 0, 0, 0, 7, 3, 0],
         [4, 2, 0, 0, 0, 0, 0, 0, 0],
         [0, 9, 0, 5, 4, 0, 0, 0, 0],
         [0, 0, 8, 9, 6, 0, 0, 5, 0]],

        [[0, 6, 0, 8, 2, 0, 0, 0, 0],
         [0, 0, 2, 0, 0, 0, 8, 0, 1],
         [0, 0, 0, 7, 0, 0, 0, 5, 0],
         [4, 0, 0, 5, 0, 0, 0, 0, 6],
         [0, 9, 0, 6, 0, 7, 0, 3, 0],
         [2, 0, 0, 0, 0, 1, 0, 0, 7],
         [0, 2, 0, 0, 0, 9, 0, 0, 0],
         [8, 0, 4, 0, 0, 0, 7, 0, 0],
         [6, 7, 0, 0, 4, 8, 0, 2, 0]]
    ]

    timecost = [list() for i in range(4)]
    nodes = [list() for i in range(4)]
    for i in range(4):
        timecost[i] = [list() for j in range(3)]
        nodes[i] = [list() for j in range(3)]
    solvers = [[SudokuSolverB(puzzle[0]), SudokuSolverBFC(puzzle[0]), SudokuSolverBFCH(puzzle[0])],
               [SudokuSolverB(puzzle[1]), SudokuSolverBFC(puzzle[1]), SudokuSolverBFCH(puzzle[1])],
               [SudokuSolverB(puzzle[2]), SudokuSolverBFC(puzzle[2]), SudokuSolverBFCH(puzzle[2])],
               [SudokuSolverB(puzzle[3]), SudokuSolverBFC(puzzle[3]), SudokuSolverBFCH(puzzle[3])]]
    for i in range(4):
        for j in range(3):
            for z in range(50):
                solvers[i][j].reinit(puzzle[i])
                solvers[i][j].solve()
                timecost[i][j].append(solvers[i][j].get_time_cost())
                nodes[i][j].append(solvers[i][j].get_node())
            cal = Stats(timecost[i][j])
            av_time = cal.avg()
            std_time = cal.stdev()
            timecost[i][j] = [av_time, std_time]
            cal = Stats(nodes[i][j])
            av_nodes = cal.avg()
            std_nodes = cal.stdev()
            nodes[i][j] = [av_nodes, std_nodes]
    print 'Time table'
    for i in timecost:
        for j in i:
            print repr(j).rjust(5),
        print

    print 'Nodes table'
    for i in nodes:
        for j in i:
            print repr(j).rjust(5),
        print

    for i in range(4):
        solvers[i][2].print_answer()
        print
