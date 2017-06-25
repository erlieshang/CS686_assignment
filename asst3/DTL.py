# Author: erlie.shang@uwaterloo.ca
import math
import copy


class Example(object):
    def __init__(self):
        self.attrs = list()
        self.classification = None


class DTN(object):
    def __init__(self, attr, classification=None):
        self.attr = attr
        self.classification = classification
        self.info_gain = None
        self.yes_branch = None
        self.no_branch = None

    @staticmethod
    def info_value(p):
        ret = 0
        if p != 0:
            ret = -p * math.log(p, 2)
        if (1-p) != 0:
            ret -= (1 - p) * math.log((1 - p), 2)
        return ret

    @staticmethod
    def choose_attr(attrs, examples):
        p = 0
        for e in examples:
            if e.classification == 1:
                p += 1
        pp = float(p) / len(examples)
        I = DTN.info_value(pp)
        max_ig = 0
        ret = None
        for a in attrs:
            p1 = 0
            n1 = 0
            p2 = 0
            n2 = 0
            for e in examples:
                if a in e.attrs:
                    if e.classification == 1:
                        p1 += 1
                    else:
                        n1 += 1
                else:
                    if e.classification == 1:
                        p2 += 1
                    else:
                        n2 += 1
            remainder = float(p1 + n1) / len(examples) * DTN.info_value(float(p1) / (p1 + n1)) + float(p2 + n2) / len(
                examples) * DTN.info_value(float(p2) / (p2 + n2))
            if I - remainder > max_ig:
                max_ig = I - remainder
                ret = a
        return [ret, max_ig] if ret is not None else attrs[0]

    @staticmethod
    def dtl(examples, attrs, depth, max_depth=100, default=None):
        if len(examples) == 0:
            return default
        elif examples[0] == examples[-1]:
            return DTN(None, examples[0].classification)
        elif len(attrs) == 0 or depth >= 100:
            return DTN(None, examples[len(examples) / 2].classification)
        else:
            best = DTN.choose_attr(attrs, examples)
            root = DTN(best[0])
            root.info_gain = best[1]
            attrs.remove(best[0])
            new_attrs = copy.deepcopy(attrs)
            new_examples = [list(), list()]
            for e in examples:
                if best in e.attrs:
                    new_examples[0].append(e)
                else:
                    new_examples[1].append(e)
            root.yes_branch = DTN.dtl(new_examples[0], attrs, depth + 1, max_depth,
                                      DTN(None, examples[len(examples) / 2].classification))
            root.no_branch = DTN.dtl(new_examples[1], new_attrs, depth + 1, max_depth,
                                     DTN(None, examples[len(examples) / 2].classification))
            return root

    def display(self):
        if self is None:
            return
        elif self.classification is not None:
            print "this is a leaf node and it's class is: ", self.classification
        else:
            print "this is an internal node and it's word feature is: ", self.attr
            print "the information gain is: ", self.info_gain
            self.yes_branch.display()
            self.no_branch.display()

    def run(self, example):
        if self.classification is not None:
            return self.classification
        if self.attr in example.attrs:
            return self.yes_branch.run(example)
        else:
            return self.no_branch.run(example)


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
            examples[current-1].attrs.append(data[1])
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
            words.append(line.split())
    return words


if __name__ == "__main__":
    train = load_data('trainData.txt', 'trainLabel.txt')
    test = load_data('testData.txt', 'testLabel.txt')
    # words = load_words()
    attrs = range(1, 3567)
    # examples = load_data('d.txt', 'l.txt')
    # attrs = range(1, 6)
    root = DTN.dtl(train, attrs, 0)
    # root.display()
    print root.run(test[0])
