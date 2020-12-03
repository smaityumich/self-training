import numpy as np
import sys
from itertools import product
import fair_self_training as fair


def f(prop = 0.1, mu = 1, index = 0):
    error_dict =  fair.biased_vs_fair_test_error(prop, mu)
    error_dict['prop'] = prop
    error_dict['separation'] = mu
    error_dict['iter-index'] = index
    return error_dict

if __name__ == "__main__":
    p = [0.01, 0.05, 0.1, 0.2, 0.4]
    mus = [2, 2.5, 3]
    parameters = list(product(p,mus, range(500)))
    i = int(float(sys.argv[1]))
    filename = sys.argv[2] 
    filename = 'files/' + filename + f'_{i}.out'  
    sub_par = parameters[(500 * i):(500 * (i+1))]
    with open(filename, 'a') as f:
        for p, mu, index in sub_par:
            error_dict = f(p, mu, index)
            f.writelines(str(error_dict) + '\n')
