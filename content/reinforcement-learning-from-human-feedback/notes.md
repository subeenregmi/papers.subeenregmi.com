https://rlhfbook.com/book.pdfa

# Introduction
- RLHF is used to incorporate human information into AI systems


---

# Training Overview

### RL Context
- Agent takes action $a_t$ sampled from policy $\pi(a_t |s_t)$ in state $s_t$ to maximize reward $r(s_t, a_t)$
- Policy $\pi$ maps state to a probability distribution over actions
- RLHF uses deep reinforcement learning policies - where $\pi$ is a deep neural network
- Transitions are probabilities from current state + current action to next state $p(s_{t+1} | s_t, a_t)$ with an initial state distribution of $p_0(s_0)$
- Policy and transitions make up a trajectory distribution
- **Trajectories overal probability** $$ p_\pi(\tau) = p_0(s_0) \prod_{t=0}^{T-1} \pi(a_t | s_t) p(s_{t+1} | a_t) $$
	- across finite horizon $T$
- **Goal of RL Agent** $$ \max_\pi \mathbb{E}_{\tau \sim p_\pi} \left[\sum_{t=0}^{T-1} \gamma^t r(s_t, a_t)\right] $$
	- maximises the future expected reward where $\gamma$ determines how far outlooking the agent is
- $J(\pi)$ indicates the expected return for a policy and $J^*(\pi)$ indicates the optimal policy.
- $T \to \infty$ and $\gamma < 1$ for well-defined objectives

### Training Objective/Loop for RLHF
$$
\max_\pi \mathbb{E}_{\tau \sim \pi}\left[r_\theta(s_t, a_t)\right]
- \beta \mathcal{D}_{KL}(\pi(\cdot | s_t) || \pi_{ref}(\cdot | s_t))
$$
where 
- $r_\theta$ is the learned reward/preference model
- $\pi_{ref}$ is a frozen snapshot of the policy/model before optimisation
- $\beta$ influence how close the model is to its starting position

We use the KL-divergence to ensure that the model does not over optimise as we are using a strong prior (the pretrained model)

### Differences between RLHF and the standard RL
1. Uses a reward model
	- Increases flexibility of approach and control for the designer
	- Model is trained on human preferences
2. No state transitions exist - goes from prompt to answer
3. Response-level reward and no discounting

### Why RL in post-training LMs?
- Can fix rough edges of the model - making it easy to talk to as an assistant
- Can be done surgically - RL does not squash the capabilities of the model

### Examples of Recipes
- **Instruct GPT** - Instruction tuning on 10K examples -> Training a reward model on 100K pairwise prompts -> Training instruction-tuned model with RLHF on a seperate 100K prompts
- Tulu 3 -> More overall prompts -> On-policy preference data on 1M preference pairs -> RLVR on 10K prompts

--- 

# Instruction Fine-Tuning
*Training a base model to answer in an user/assistant format*

**Chat template** - the format of interaction between user/assistant that the model must conform to
- Three roles: 
	- system - system prompt
	- user
	- assistant

### Best practices
- High quality data
- Around 1M prompts
- Similar distribution to downstream tasks of interest
- Training after instruction tunign allows models to receover from some noise

### Implementation details
- **Smaller batch sizes** - allows models to optimize to a narrower data distribution whilst preserving models generalization
- **Prompt masking** - prompt tokens are masked out so model does not learn to predict the user queries
- **Multi-turn masking** 
	1. Final turn only - only tokens in final assistant turn is included in the loss
	2. Mask user turn only - all user tokens masked, loss includes all assistant turns
- **Same loss as pretraining** 
- **Lower learning rate**

--- 
# Reward Modelling
*Constructing a model to learn human preferences in prompts*

#### Notation
- $x$ - prompts
- $y$ - completions

**Bradley-Terry model of preference** - Probability that in a pairwise comparison between two items $i$ and $j$, a judge prefers $i$ over $j$
$$
P(i < j) = \frac{p_i}{p_i + p_j}
$$
- assumes that each item has $p_i > 0$
- often is reperametrized where $p_i = e^{r_i}$
$$
P(i < j) = \frac{e^{r_i}}{e^{r_i} + e^{r_j}} = \sigma(r_i - r_j)
$$
	- where $\sigma$ is the sigmoid function

### Loss function
Reward model is trained on a loss function based on the Bradley-Terry relation. To do so it is given a small linear layer to be able to predict a scalar score.

Given prompt $x$ and two sampled completions $y_c$ and $y_r$, where $y_c$ is the chosen completion and $y_r$ is the rejected one, the loss function per example is
$$
\large
\mathcal{L(\theta)} = \log \left(1 + e^{r_\theta(y_r \mid x) - r_\theta(y_c \mid x)}\right)
$$

When training language models only one epoch is completed to prevent overfitting.

## Outcome Reward Models
*Learning a per-token signal of how likely the completion is to end in a correct answer over time*

Here we are given a prompt $x$ and our inductive bias is that one completion should be correct whereas the others are incorrect, $(y_c, y_{ic})$. The ORM is training a per-token predictor of if an answer is correct. The per-token  loss function applies the binary cross-entropy at each completion token, where each token's outcome probability is trained towards the sequence's outcome label.
$$
\large
\mathcal{L}_\text{token}(\theta) = - \mathbb{E}_{(s, r) \sim \mathcal{D}}\left[\frac{1}{T} \sum_{t=1}^T (r \log p_\theta(s_t) + (1 - r) \log(1 - p_\theta(s_t)))\right]
$$
where
- $s$ is a completion of $T$ tokens
- $r \in \{0, 1\}$ - binary label of correctness
- $p_\theta(s_t) = \sigma(w_\theta(s_t))$ - probability of correctness predicted at token $t$ from model's scalar logit $w_\theta(s_t)$

## Process Reward Models
*Reward models trained to output scores at every step in a chain-of-though reasoning process*

PRMs is often optimized with a per-step cross-entropy loss:
![[Screenshot 2026-08-24 at 07.50.49.png]]
where
- $s$ is a sampled chain-of-thought with $K$ annotated steps
- $y_{s_i} \in \{0, 1\}$ denotes whether $i$-th step is correct
- $r_\theta(s_i | x , s_{< i})$ - PRMs predicted probability that step $s_i$ is valid condition on prompt $x$ and all previous steps $s_{<i}$ 