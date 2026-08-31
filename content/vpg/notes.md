- Prior to this method, value function approach is where the effort goes into estimating a value function with a greedy policy approach
	- Works well for deterministic policies
	- Bad as small changes in estimated value changes the action selected
- Paper approximates a stochastic policy directly using a independent fucntion approximator with its own parameters
	- Inputs could be state reprenstation, output is action probabilities and weights are the policy parameters

**Policy Gradiant Approach** - Policy parameters are updated proportional yo the graidient
$$
\Delta\theta \approx \alpha \frac{\partial \rho}{ \partial \theta}
$$
where 
- $\rho$ - performance of the policy (average reward per step)
- $\theta$ - vector of policy parameters
- $\alpha$ - positive step size
- 

