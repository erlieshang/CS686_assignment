# Author: erlie.shang@uwaterloo.ca
import Queue
import math


class MLL(object):
    def __init__(self, num):
        self.classification = None
        # ignore the first one
        self.parameters = [[0, 0] for i in range(num)]

    def learn(self, examples):
        num_of_class_one = 0
        for e in examples:
            if e.classification == 1:
                num_of_class_one += 1
            for a in e.attrs:
                self.parameters[a][e.classification - 1] += 1
        for p in self.parameters:
            p[0] = float(p[0] + 1) / (num_of_class_one + 1)
            p[1] = float(p[1] + 1) / ((len(examples) - num_of_class_one) + 1)

    def run(self, data):
        p1 = float(0)
        p2 = float(0)
        for p in range(1, len(self.parameters)):
            if p in data.attrs:
                p1 += math.log(self.parameters[p][0])
                p2 += math.log(self.parameters[p][1])
            else:
                p1 += math.log(1 - self.parameters[p][0])
                p2 += math.log(1 - self.parameters[p][1])
        return 1 if p1 > p2 else 2

    def top10(self):
        ret = list()
        q = Queue.PriorityQueue()
        for i in range(1, len(self.parameters)):
            d = abs(math.log(self.parameters[i][0]) - math.log(self.parameters[i][1]))
            q.put((d, words[i]))
            if q.qsize() > 10:
                q.get()
        while not q.empty():
            ret.append(q.get())
        return ret


class Example(object):
    def __init__(self):
        self.attrs = list()
        self.classification = None

def load_data(data, label):
    examples = list()
    with open(data, 'r') as f:
        current = 1
        examples.append(Example())
        for line in f:
            data = [int(x) for x in line.split()]
            while data[0] != current:
                current += 1
                examples.append(Example())
            examples[current - 1].attrs.append(data[1])
    with open(label, 'r') as f:
        index = 0
        for line in f:
            examples[index].classification = int(line.split()[0])
            index += 1
    return examples


def load_words():
    words = list()
    words.append('')
    with open('words.txt', 'r') as f:
        for line in f:
            words.append(line.split()[0])
    return words


if __name__ == "__main__":
    train = load_data('trainData.txt', 'trainLabel.txt')
    test = load_data('testData.txt', 'testLabel.txt')
    words = load_words()
    x = MLL(len(words))
    x.learn(train)
    train_accuracy = 0
    test_accuracy = 0
    for e in train:
        if x.run(e) == e.classification:
            train_accuracy += 1
    train_accuracy = float(train_accuracy) / len(train)
    for e in test:
        if x.run(e) == e.classification:
            test_accuracy += 1
    test_accuracy = float(test_accuracy) / len(test)
    print "The train accuracy is: ", train_accuracy
    print "The test accuracy is: ", test_accuracy
    top10 = x.top10()
    top10.reverse()
    for e in top10:
        print e
