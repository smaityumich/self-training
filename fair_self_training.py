import numpy as np
from sklearn.linear_model import LogisticRegression
import utils
from itertools import product
import sys

def biased_vs_fair_test_error(prop, mu,  n_biased = 60, n_fair = 2000):
    
    
    # Generating biased data
    # Labeled biased data
    biased_train_x, biased_train_y = utils.sim_x_y(prop, n = n_biased, mu = mu)

    # Unlableled fair data and test data
    fair_train_x, fair_train_y = utils.sim_x_y(0.5, n = n_fair, mu=mu)
    fair_test_x,  fair_test_y = utils.sim_x_y(0.5, n = 2000, mu=mu)
    # Baseline logistic classifier
    cl_unfair = LogisticRegression(max_iter=1000)
    cl_unfair.fit(biased_train_x, biased_train_y)
    baseline_fair = np.mean(np.absolute(fair_test_y - cl_unfair.predict(fair_test_x)))
    
    

    # Self-trained logistic classifier
    cl_self_trained = cl_unfair
    for i in range(100):
        pseudo_fair_train_y = cl_self_trained.predict(fair_train_x)
        augmented_x = np.concatenate([biased_train_x, fair_train_x], axis = 0)
        augmented_y = np.concatenate([biased_train_y, pseudo_fair_train_y], axis = 0)
        cl_self_trained.fit(augmented_x, augmented_y)
    self_trained_fair = np.mean(np.absolute(fair_test_y - cl_self_trained.predict(fair_test_x)))
    

    errors = dict()
    errors['baseline-error'] = baseline_fair
    errors['self-trained-error'] = self_trained_fair
    return errors




