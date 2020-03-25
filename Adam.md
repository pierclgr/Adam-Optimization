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

> *m = β1 m + (1 - β1) g*
> *v = β2 v + (1 - β2) |g|^2*

m and v are estimates of the first moment (the mean) and the second moment (the uncentered variance) of the gradients respectively.
As m and v are initialized as vectors of 0's, the authors of Adam observe that they are biased towards zero, especially during the initial time steps.
They counteract these biases by computing bias-corrected first and second moment estimates:

> *m_hat = m / (1 - β1^t)*
> *v_hat = v / (1 - β2^t)*

Once estimators are calculated and corrected, the parameters are updated using the following formula:

> *theta = theta - alpha * m_hat / (√v + ϵ)*

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
v(v_hat).

## Nadam
It implements the Nesterov accelerated momentum using it as m. It basically updates the parameters with the momentum step before computing the gradient.

## Amsgrad

In some cases, e.g. for object recognition or machine translation they fail to converge to an optimal solution and are outperformed by SGD with momentum.

Reddi et al. (2018) formalize this issue and pinpoint the exponential moving average of past squared gradients as a reason for the poor generalization behaviour of adaptive learning rate methods. Recall that the introduction of the exponential average was well-motivated: it should prevent the learning rates to become infinitesimally small as training progresses
However, this short-term memory of the gradients becomes an obstacle in other scenarios. 

In settings where Adam converges to a suboptimal solution, it has been observed that some minibatches provide large and informative gradients, but as these minibatches only occur rarely, exponential averaging diminishes their influence, which leads to poor convergence.

To fix this behaviour, the authors propose a new algorithm, AMSGrad that uses the maximum of past squared gradients v rather than the exponential average to update the parameters. 
v is defined the same as in Adam above:

> *v = β2 v + (1 - β2) |g|^2*

Instead of using v(or its bias-corrected version v_hat) directly, we now employ the previous v(t-1) if it is larger than the current one:

> *v_hat = max(v_hat−1, vt)*

This way, AMSGrad results in a non-increasing step size, which avoids the problems suffered by Adam. For simplicity, the authors also remove the debiasing step that we have seen in Adam.

> *m = β1 m + (1 - β1) g*
> *v = β2 v + (1 - β2) |g|^2*
> *v_hat = max(v_hat−1, vt)*
> *theta = theta - alpha * m / (√v + ϵ)*

The authors observe improved performance compared to Adam on small datasets and on CIFAR-10






