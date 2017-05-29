import numpy as np


class Factor(object):
    def __init__(self, data=[], var_list=[]):
        self.array = np.array(data)
        self.var = var_list


def restrict(factor, variable, value):
    if len(factor.var) < 2:
        return factor
    if variable in factor.var:
        index = factor.var.index(variable)
        factor.var.remove(variable)
        x = 0 if value else 1
        slc = [slice(None)] * len(factor.var)
        slc[index] = x
        return Factor(factor.array[slc], factor.var)


def multiply(f1, f2):
    var_list = list(f1.var)
    for v in f2.var:
        if v not in var_list:
            var_list.append(v)
    var_list.sort()
    shape = get_shape(f1.var, var_list)
    f1.array.resize(shape)
    shape = get_shape(f2.var, var_list)
    f2.array.resize(shape)
    return Factor(f1.array * f2.array, var_list)


def get_shape(var, combined_var):
    shape = [2 for i in range(len(combined_var))]
    i = 0
    j = 0
    while j < len(combined_var):
        if i >= len(var) or combined_var[j] != var[i]:
            shape[j] = 1
        else:
            i += 1
        j += 1
    return shape


def sumout(factor, variable):
    if len(factor.var) < 2:
        return factor
    if variable in factor.var:
        index = factor.var.index(variable)
        factor.var.remove(variable)
        return Factor(factor.array.sum(axis=index), factor.var)


def normalize(factor):
    sum = factor.array.sum()
    factor.array /= sum


def inference(factorList, queryVariables, orderedListOfHiddenVariables, evidenceList = []):
    nfl = list()
    if type(evidenceList) == dict:
        for f in factorList:
            tmp = Factor()
            for (k, v) in evidenceList.items():
                if k in f.var:
                    tmp = restrict(f, k, v)
            nfl.append(tmp)
    else:
        nfl = factorList
    ret = nfl[0]
    for i in range(1, len(nfl)):
        ret = multiply(ret, nfl[i])
    for i in orderedListOfHiddenVariables:
        ret = sumout(ret, i)
        if ret.var == queryVariables:
            break
    normalize(ret)
    return ret

if __name__ == "__main__":
    order = ['Trav', 'FP', 'Fraud', 'IP', 'OC', 'CRP']

    #q2 (b) Pr(Fraud)
    f_Tr = Factor([0.05, 0.95], ['Trav'])
    f_Fr_Tr = Factor([[0.01, 0.004], [0.99, 0.996]], ['Fraud', 'Trav'])
    x = inference([f_Tr, f_Fr_Tr], 'Fraud', ['Trav'])
    print(x.array, x.var)

    # q2 (b) Pr(Fraud|fp, ~ip, crp)
    # f1(OC, Fraud) = Pr(~ip|OC, Fraud)
    f1 = Factor([[0.85, 0.049], [0.9, 0.999]], ['Fraud', 'OC'])
    # f2(OC) = Pr(crp|OC)
    f2 = Factor([0.1, 0.01], ['OC'])
    # f3(OC) = Pr(OC)
    f3 = Factor([0.8, 0.2], ['OC'])
    # f4(Trav, Fraud) = Pr(fp|Trav, Fraud)
    f4 = Factor([[0.9, 0.1], [0.9, 0.01]], ['Fraud', 'Trav'])
    # f5(Trav) = Pr(Trav)
    f5 = Factor([0.05, 0.95], ['Trav'])
    # f6(Fraud) = Pr(Fraud)
    f6 = Factor([0.0043, 0.9957], ['Fraud'])
    x = inference([f1,f2,f3,f4,f5,f6], 'Fraud', ['Trav', 'OC'])
    print(x.array, x.var)

    # q2 (c) Pr(Fraud|fp, ~ip, crp, trav)
    # f1(OC, Fraud) = Pr(~ip|OC, Fraud)
    f1 = Factor([[0.85, 0.049], [0.9, 0.999]], ['Fraud', 'OC'])
    # f2(OC) = Pr(crp|OC)
    f2 = Factor([0.1, 0.01], ['OC'])
    # f3(OC) = Pr(OC)
    f3 = Factor([0.8, 0.2], ['OC'])
    # f4(Fraud) = Pr(fp|trav, Fraud)
    f4 = Factor([[0.9, 0.9]], ['Fraud'])
    # f6(Fraud) = Pr(Fraud)
    f6 = Factor([0.0043, 0.9957], ['Fraud'])
    x = inference([f1, f2, f3, f4, f6], 'Fraud', ['OC'])
    print(x.array, x.var)


    # q2 (d) Pr(Fraud|ip)
    # f1(OC, Fraud) = Pr(ip|Fraud)
    f1 = Factor([[0.15, 0.051], [0.1, 0.001]], ['Fraud', 'OC'])
    # f2(Fraud) = Pr(Fraud)
    f2 = Factor([0.0043, 0.9957], ['Fraud'])
    x = inference([f1, f2], 'Fraud', ['OC'])
    print(x.array, x.var)
    # f3(OC) = Pr(crp|OC)
    f3 = Factor([0.1, 0.01], ['OC'])
    x = inference([f1, f2, f3], 'Fraud', ['OC'])
    print(x.array, x.var)