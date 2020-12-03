import numpy as np
from compas_data import get_compas_train_test
from sklearn.linear_model import LogisticRegression
from fairlearn.reductions import ExponentiatedGradient
from fairlearn.reductions import EqualizedOdds
from utils import sample_balanced, reductions_prob

## Prepare data
np.random.seed(1)
x_train, x_test, y_train, y_test, y_sex_train, \
y_sex_test, y_race_train, y_race_test, feature_names = get_compas_train_test(pct=0.7)
group_train_cross = y_sex_train + 2*y_race_train
group_test_cross = y_sex_test + 2*y_race_test

## Reductions NO fairness
eps_base = 10.
constraint = EqualizedOdds()
classifier = LogisticRegression(solver='liblinear', fit_intercept=True)
mitigator_base = ExponentiatedGradient(classifier, constraint, eps=eps_base, T=50)
mitigator_base.fit(x_train, y_train, sensitive_features=group_train_cross)
classifier_base = lambda x: (reductions_prob(mitigator_base, x)>0.5)*1

## Reductions fair classifier
eps = 0.1
constraint = EqualizedOdds()
classifier = LogisticRegression(solver='liblinear', fit_intercept=True)
mitigator = ExponentiatedGradient(classifier, constraint, eps=eps, T=50)
mitigator.fit(x_train, y_train, sensitive_features=group_train_cross)
classifier_fair = lambda x: (reductions_prob(mitigator, x)>0.5)*1

acc_fair = 0
acc_base = 0
n_trial = 50
for i in range(n_trial):
    idx = sample_balanced(y_test, group_test_cross)
    acc_trial_base = (classifier_base(x_test[idx]) == y_test[idx]).mean()
    acc_base += acc_trial_base
    acc_trial_fair = (classifier_fair(x_test[idx]) == y_test[idx]).mean()
    acc_fair += acc_trial_fair

acc_base = acc_base/n_trial
acc_fair = acc_fair/n_trial

acc_biased_test_base = (classifier_base(x_test) == y_test).mean()
acc_biased_test_fair = (classifier_fair(x_test) == y_test).mean()

## Check accuracy
print('Test accuracy on P-tilde')
print('Base', acc_biased_test_base)
print('Fair', acc_biased_test_fair)

print('\nTest accuracy on P-star')
print('Base', acc_base)
print('Fair', acc_fair)