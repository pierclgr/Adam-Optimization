## Stochastic Gradient Descent
The formula used to update the parameters θ is
**θ = θ - ∇J(θ)**
* **Gradient Descent** updates the weights θ using the gradient computer in the entire training set once
* **Stochastic Gradient Descent** updates the weights computing the gradient once for each training sample, performing one update at a time
    * **Minibatch Gradient Descent** updates the weights computing the gradients for multiple subsets of the training set

The disadvantage of SGD is the fact that updating the parameters frequently produces high fluctuations, complicating the convergence to the exact minimum. Fluctuations also causes the algorithm to continuously overshoot.

## Adam
Adaptive Moment Estimation (Adam) is a method that computes adaptive learning rates for each parameter. It stores an exponentially decaying average of past squared gradients 
v and Adam also keeps an exponentially decaying average of past gradients m, similar to momentum.  We compute the decaying averages of past and past squared gradients m and v respectively as follows: 
![alt text](https://miro.medium.com/max/886/1*ZhGLUwaaqlJ9C0WK0nbAEA.png)
m and v are estimates of the first moment (the mean) and the second moment (the uncentered variance) of the gradients respectively.
As m and v are initialized as vectors of 0's, the authors of Adam observe that they are biased towards zero, especially during the initial time steps.
They counteract these biases by computing bias-corrected first and second moment estimates:
![alt text](https://miro.medium.com/max/390/1*M86IUMsrHXq4WrS-Bk5boA.png)
Once estimators are calculated and corrected, the parameters are updated using the following formula:
![alt text](https://miro.medium.com/max/520/1*tKn5TEW-7aQoerAeDB8x6g.png)
β1, β2 and ϵ are hyperparameters like alpha (the learning rate), ϵ  is a small scalar used to prevent division by 0 and β1 and β2 control exponential decay.
The authors propose default values of 0.9 for β1, 0.999 for β2, and 10^(−8) for ϵ
Since Adam is derived from SGD, these operations are perfomed for each training sample.

## Adamax

The v factor in the Adam update rule scales the gradient inversely proportionally to the ℓ2 norm of the past gradients (via the v(t−1) term) and current gradient |g|^2:

> *v = β2 v + (1 - β2) |g|^2*

We can generalize this update to the ℓp norm:

> *v = β2^p v + (1 - β2^p) |g|^p*

Norms for large  p values generally become numerically unstable, which is why ℓ1 and ℓ2 norms are most common in practice. However, ℓ∞ also generally exhibits stable behavior. For this reason, the authors propose AdaMax and show that v with ℓ∞ converges to the following more stable value. 

> *v = β2^∞ v + (1 - β2^∞) |g|^∞ = max(β2 v, |g|)*

We can now plug this into the Adam update equation by replacing √v + ϵ to obtain the AdaMax update rule:

> *theta = theta - alpha * m_hat / v*

Note that as v now relies on the max operation, it is not as suggestible to bias towards zero as m and v in Adam, which is why we do not need to compute a bias correction for 
v.

## Nadam
It implements the Nesterov accelerated momentum using it as m. It basically updates the parameters with the momentum step before computing the gradient.

## Amsgrad
It uses the maximum value of v instead uf using the moving average.