## Stochastic Gradient Descent
The formula used to update the parameters θ is
**θ = θ - ∇J(θ)**
* **Gradient Descent** updates the weights θ using the gradient computer in the entire training set once
* **Stochastic Gradient Descent** updates the weights computing the gradient once for each training sample, performing one update at a time
    * **Minibatch Gradient Descent** updates the weights computing the gradients for multiple subsets of the training set

The disadvantage of SGD is the fact that updating the parameters frequently produces high fluctuations, complicating the convergence to the exact minimum. Fluctuations also causes the algorithm to continuously overshoot.

## Adam
Adam uses two estimators called m and v which are respectively the mean and the uncentered variance of the gradients. They are also called first and second momentum.
![alt text](https://miro.medium.com/max/886/1*ZhGLUwaaqlJ9C0WK0nbAEA.png)
Estimators are then *bias corrected*, so two new estimators called m_hat and v_hat are calculated. These are the bias corrected estimators: since we had, in the first iteraction, the estimator initialized to zero, the estimators are biased towards zero. So estimators need to be corrected. This process is called bias correction.
![alt text](https://miro.medium.com/max/390/1*M86IUMsrHXq4WrS-Bk5boA.png)
Once estimators are calculated and corrected, the parameters are updated using the following formula:
![alt text](https://miro.medium.com/max/520/1*tKn5TEW-7aQoerAeDB8x6g.png)
Beta1, Beta2 and epsilon are hyperparameters like alpha (the learning rate).
Since Adam is derived from SGD, these operations are perfomed for each training sample.

## Adamax
Is similar to Adam but the estimator v is seen as the L2 norm (the distance) between the current gradient and the past gradients.

## Nadam
It implements the Nesterov accelerated momentum using it as m. It basically updates the parameters with the momentum step before computing the gradient.

## Amsgrad
It uses the maximum value of v instead uf using the moving average.