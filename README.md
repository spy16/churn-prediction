# Chrun Prediction

Churn prediction exercise using Deep Neural Network.

## Problem

Given various information about current and exited customers of a certain bank,
predict wether a current customer is likely to churn.

Customer data is given as a CSV file: `customer-data.csv`

## Solution

Solution is based on a deep neural network that outputs the probability of churn (a scalar).

1. `predict_churn_v1.py`:

    * Architecture: (11) ==> (6, ReLU) ==> (6, ReLU) ==> (1, Sigmoid)
    * Hyper parameter tuning: manual
    * Accuracy (train): 84.09%
    * Accuracy (test) : 82.13%

2. `predict_churn_v2.py`:

    * Architecture: (11) ==> Dense(6, ReLU) ==> Dropout@0.2 ==> (6, ReLU)  ==> Dropout@0.2 ==> (1, Sigmoid)
    * Hyper parameter tuning: manual
    * Accuracy (train): 82.92857142857143%
    * Accuracy (test) : 82.23333333333333%

3. `predict_churn_v3.py`:

    * Architecture: (11) ==> Dense(6, ReLU) ==> Dropout@0.2 ==> (6, ReLU)  ==> Dropout@0.2 ==> (1, Sigmoid)
    * Hyper parameter tuning: manual
    * k-Fold Cross Validation: Max=0.860000 Min=0.774286 [Mean=0.802714 Variance=0.022619]
    * Accuracy (train): 79.27142857142857%
    * Accuracy (test) : 80.46666666666667%
