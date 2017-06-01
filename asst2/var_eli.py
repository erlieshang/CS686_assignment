import numpy as np
import copy


class Factor(object):
    def __init__(self, data=[], var_list=[]):
        self.array = np.array(data)
        self.var = var_list


def restrict(factor, variable, value):
    if len(factor.var) < 2:
        return False
    if variable in factor.var:
        index = factor.var.index(variable)
        factor.var.remove(variable)
        x = 0 if value else 1
        return_src = 'Factor(factor.array[%s %d, ...], factor.var)' % (':,' * index, x)
        return eval(return_src)


def multiply(f1, f2):
    x1 = copy.deepcopy(f1)
    x2 = copy.deepcopy(f2)
    var_list = list(x1.var)
    for v in x2.var:
        if v not in var_list:
            var_list.append(v)
    var_list.sort()
    shape = get_shape(x1.var, var_list)
    x1.array.resize(shape)
    shape = get_shape(x2.var, var_list)
    x2.array.resize(shape)
    return Factor(x1.array * x2.array, var_list)


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
    # restrict
    if type(evidenceList) == dict:
        for f in factorList:
            tmp = copy.deepcopy(f)
            for (k, v) in evidenceList.items():
                if k in tmp.var:
                    tmp = restrict(tmp, k, v)
            if tmp:
                nfl.append(tmp)
                print 'restrict', k, tmp.array, tmp.var
    else:
        nfl = copy.deepcopy(factorList)

    # sumout the product of relevant factors
    for v in orderedListOfHiddenVariables:
        vlist = list()
        for f in nfl:
            if v in f.var:
                vlist.append(f)
        for f in vlist:
            nfl.remove(f)
        new_f = copy.deepcopy(vlist[0])
        for i in range(1, len(vlist)):
            new_f = multiply(new_f, vlist[i])
        new_f = sumout(new_f, v)
        print 'sumout variable', v, new_f.array, new_f.var
        nfl.append(new_f)

    # multiply the sumouted factors
    ret = nfl[0]
    for i in range(1, len(nfl)):
        ret = multiply(ret, nfl[i])

    # normalize
    normalize(ret)
    return ret

if __name__ == "__main__":
    order = ['Trav', 'Fp', 'Fraud', 'Ip', 'OC', 'Crp']
    # f1(OC) = Pr(OC)
    f1 = Factor([0.8, 0.2], ['OC'])
    # f2(Trav) = Pr(Trav)
    f2 = Factor([0.05, 0.95], ['Trav'])
    # f3(Fraud, Trac) = Pr(Fraud|Trav)
    f3 = Factor([[0.01, 0.004], [0.99, 0.996]], ['Fraud', 'Trav'])
    # f4(Fp, Trav, Fraud) = Pr(Fp|Trav, Fraud)
    f4 = Factor([[[0.9, 0.1], [0.9, 0.01]], [[0.1, 0.9], [0.1, 0.99]]], ['Fp', 'Fraud', 'Trav'])
    # f5(Ip, OC, Fraud) = Pr(Ip|OC, Fraud)
    f5 = Factor([[[0.15, 0.051], [0.85, 0.949]], [[0.1, 0.001], [0.9, 0.999]]], ['Fraud', 'Ip', 'OC'])
    # f6(Crp, OC) = Pr(Crp|OC)
    f6 = Factor([[0.1, 0.01], [0.9, 0.99]], ['Crp', 'OC'])

    f_list = [f1, f2, f3, f4, f5, f6]

    #q2 (b) Pr(Fraud)
    print 'Pr(Fraud)'
    h_order = ['Trav', 'Fp', 'Ip', 'OC', 'Crp']
    x = inference(f_list, 'Fraud', h_order)
    print (x.array, x.var)

    #q2 (b) Pr(Fraud|fp, ~ip, crp)
    print 'Pr(Fraud|fp, ~ip, crp)'
    h_order = ['Trav', 'OC']
    x = inference(f_list, 'Fraud', h_order, {'Fp': True, 'Ip': False, 'Crp': True})
    print (x.array, x.var)

    # q2 (c) Pr(Fraud|fp, ~ip, crp, trav)
    print 'Pr(Fraud|fp, ~ip, crp, trav)'
    h_order = ['OC']
    x = inference(f_list, 'Fraud', h_order, {'Fp': True, 'Ip': False, 'Crp': True, 'Trav': True})
    print (x.array, x.var)

    # q2 (d)
    # Pr(Fraud|ip)
    print 'Pr(Fraud|ip)'
    h_order = ['Trav', 'Fp', 'OC', 'Crp']
    x = inference(f_list, 'Fraud', h_order, {'Ip': True})
    print (x.array, x.var)

    # Pr(Fraud|ip, crp)
    print 'Pr(Fraud|ip, crp)'
    h_order = ['Trav', 'Fp', 'OC']
    x = inference(f_list, 'Fraud', h_order, {'Ip': True, 'Crp': True})
    print (x.array, x.var)