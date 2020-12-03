import numpy as np
from utils import simul_x_y_a, reductions_prob, plot_decision_ax
from sklearn.linear_model import LogisticRegression
from fairlearn.reductions import ExponentiatedGradient, EqualizedOdds
import matplotlib.pyplot as plt
    
np.random.seed(1)

mu_mult = 2.
cov_mult = 1.
skew = 5.

## Code follows the y,a order in indexing

train_prop_mtx = [[0.4, 0.1],[0.4, 0.1]]
train_x, train_a, train_y = simul_x_y_a(train_prop_mtx, n=1000, mu_mult=mu_mult, 
                                        cov_mult=cov_mult, skew=skew)

test_prop_mtx = [[0.25, 0.25],[0.25, 0.25]]
test_x, test_a, test_y = simul_x_y_a(test_prop_mtx, n=1000, mu_mult=mu_mult, 
                                     cov_mult=cov_mult, skew=skew)

test_biased_x, test_biased_a, test_biased_y = simul_x_y_a(train_prop_mtx, n=1000, 
                                                          mu_mult=mu_mult, cov_mult=cov_mult,
                                                          skew=skew)

## Reductions NO fairness
eps_base = 10.
constraint = EqualizedOdds()
classifier = LogisticRegression(solver='liblinear', fit_intercept=True)
mitigator_base = ExponentiatedGradient(classifier, constraint, eps=eps_base, T=25)
mitigator_base.fit(train_x, train_y, sensitive_features=train_a)
base_classifier = lambda x: reductions_prob(mitigator_base, x)

## Reductions Fair classifier
eps = 0.1
constraint = EqualizedOdds()
classifier = LogisticRegression(solver='liblinear', fit_intercept=True)
mitigator = ExponentiatedGradient(classifier, constraint, eps=eps, T=25)
mitigator.fit(train_x, train_y, sensitive_features=train_a)
fair_classifier = lambda x: reductions_prob(mitigator, x)

## PLOT
all_data = np.vstack([train_x, test_x])
x_min, x_max = all_data[:, 0].min() - .5, all_data[:, 0].max() + .5
y_min, y_max = all_data[:, 1].min() - .5, all_data[:, 1].max() + .5
max_vals = [x_min, y_min, x_max, y_max]

fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(28, 7))
plot_decision_ax(axes[0], train_x, train_a, train_y, max_vals, base_classifier, y_ticks=True)
plot_decision_ax(axes[1], train_x, train_a, train_y, max_vals, fair_classifier, y_ticks=False)
plot_decision_ax(axes[2], test_x, test_a, test_y, max_vals, base_classifier, y_ticks=False)
plot_decision_ax(axes[3], test_x, test_a, test_y, max_vals, fair_classifier, y_ticks=False)
fig.tight_layout()
plt.savefig('simul_decision.pdf', bbox_inches='tight')
plt.show()

## Check accuracy
print('Test accuracy on P-tilde')
print('Base', ((base_classifier(test_biased_x) > 0.5) == test_biased_y).mean())
print('Fair', ((fair_classifier(test_biased_x) > 0.5) == test_biased_y).mean())

print('\nTest accuracy on P-star')
print('Base', ((base_classifier(test_x) > 0.5) == test_y).mean())
print('Fair', ((fair_classifier(test_x) > 0.5) == test_y).mean())