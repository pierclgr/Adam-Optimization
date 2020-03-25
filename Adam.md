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
Is similar to Adam but the estimator v is seen as the L2 norm (the distance) between the current gradient and the past gradients.

## Nadam
It implements the Nesterov accelerated momentum using it as m. It basically updates the parameters with the momentum step before computing the gradient.

## Amsgrad
It uses the maximum value of v instead uf using the moving average.