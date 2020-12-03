import numpy as np
import sys
from itertools import product
import fair_self_training as fair




if __name__ == "__main__":
    p = np.arange(0.1, 0.4, 0.05)
    mus = [2, 2.5, 3, 3.5, 4, 4.5, 5]
    parameters = list(product(p,mus, range(500)))
    i = int(float(sys.argv[1]))
    filename = sys.argv[2] 
    filename = 'files/' + filename + f'_{i}.out'  
    sub_par = parameters[(50 * i):(50 * (i+1))]
    print(sub_par)
    with open(filename, 'a') as f:
        for p, mu, index in sub_par:
            error_dict =  fair.biased_vs_fair_test_error(p, mu)
            error_dict['p'] = p
            error_dict['mu'] = mu
            error_dict['iter'] = index
            f.writelines(str(error_dict) + '\n')
