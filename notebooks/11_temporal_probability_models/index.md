# Temporal Probability Models

## Contents
- [Filtering](#filtering)
	- [An Example](#an-example)
- [Prediction](#prediction)
- [Smoothing](#smoothing)
	- [An Example](#an-example-1)
- [Most likely explanation](#most-likely-explanation)
	- [Recall: The Hidden Markov Model](#recall-the-hidden-markov-model)
	- [Likelihood Computation: The Forward Algorithm](#likelihood-computation-the-forward-algorithm)
	- [Decoding: The Viterbi Algorithm](#decoding-the-viterbi-algorithm)
	- [HMM Training: The Forward-Backward Algorithm](#hmm-training-the-forward-backward-algorithm)
- [Resources](#resources)

## Filtering
Filtering is the task of computing the **belief state** which is the posterior distribution over the most recent state, given all evidence to date. Filtering is also called state estimation. We wish to compute $P(X_t | e_{1:t})$. 
| ![Umbrella Example](./assets/umb-ex.jpg) | 
|:--:| 
| *Bayesian network structure and conditional distributions describing the umbrella world.* |
In the umbrella example, this would mean computing the probability of rain today, given all the observations of the umbrella carrier made so far. Filtering is what a rational agent does to keep track of the current state so that rational decisions can be made. It turns out that an almost identical calculation provides the likelihood of the evidence sequence, $P(e_{1:t})$.

A useful filtering algorithm needs to maintain a current state estimate and update it, rather than going back over the entire history of percepts for each update. (Otherwise, the cost of each update increases as time goes by.) In other words, given the result of filtering up to time t, the agent needs to compute the result for $t + 1$ from the new evidence $e_{t+1}$,
$$
P(X_{t+1} | e_{1:t+1}) = f(e_{t+1}, P(X_t | e_{1:t})) ,
$$
for some function $f$. This process is called **recursive estimation**. We can view the calculation as being composed of two parts: first, the current state distribution is projected forward from $t$ to $t+1$; then it is updated using the new evidence $e_{t+1}$. This two-part process emerges quite simply when the formula is rearranged:
$$
\begin{align*}
P(X_{t+1} | e_{1:t+1}) &= P(X_{t+1} | e_{1:t}, e_{t+1}) \quad \text{(dividing up the evidence)} \\
&= \alpha P(e_{t+1} |X_{t+1}, e_{1:t}) P(X_{t+1} | e_{1:t}) \quad \text{(using Bayes’ rule)} \\
&= \alpha P(e_{t+1} |X_{t+1}) P(X_{t+1} | e_{1:t}) \quad \text{(by the sensor Markov assumption).}
\end{align*}
$$
Here $\alpha$ is a normalizing constant used to make probabilities sum up to 1. The second term, $P(X_{t+1} | e_{1:t})$ represents a one-step prediction of the next state, and the first term updates this with the new evidence; notice that $P(e_{t+1} |X_{t+1})$ is obtainable directly from the sensor model.
Now we obtain the one-step prediction for the next state by conditioning on the current state $X_t$:
$$
\begin{align*}
P(X_{t+1} | e_{1:t+1}) &= \alpha P(e_{t+1} |X_{t+1}) \sum_{x_t} P(X_{t+1} | x_t, e_{1:t})P(x_t | e_{1:t})  \\
&= \alpha P(e_{t+1} |X_{t+1}) \sum_{x_t} P(X_{t+1} | x_t)P(x_t | e_{1:t}) \quad \text{(Markov assumption).}
\end{align*}
$$
Within the summation, the first factor comes from the transition model and the second comes from the current state distribution. Hence, we have the desired recursive formulation. We can think of the filtered estimate $P(X_t | e_{1:t})$ as a "message" $f_{1:t}$ that is propagated forward along the sequence, modified by each transition and updated by each new observation. The process is given by
$$
f_{1:t+1} = \alpha \text{FORWARD}(f_{1:t}, e_{t+1}) ,
$$
where FORWARD implements the update described in previous equation and the process begins with $f_{1:0} = P(X_0)$. When all the state variables are discrete, the time for each update is constant (i.e., independent of t), and the space required is also constant.

### An Example
Let us illustrate the filtering process for two steps in the basic umbrella example. That is, we will compute $P(R_2 | u_{1:2})$ as follows:

- On day 0, we have no observations, only the security guard’s prior beliefs; let’s assume that consists of $P(R_0) = <0.5, 0.5>$.
- On day 1, the umbrella appears, so $U_1 =true$. The prediction from $t=0$ to $t=1$ is
$$
\begin{align*}
P(R_1) &= \sum_{r_0} P(R_1 | r_0)P(r_0) \\
& = <0.7, 0.3> \times 0.5 + <0.3, 0.7> \times 0.5 = <0.5, 0.5> .
\end{align*}
$$
Then the update step simply multiplies by the probability of the evidence for $t=1$ and normalizes:
$$
\begin{align*}
P(R_1 | u_1) &= \alpha P(u_1 |R_1)P(R_1) = \alpha <0.9, 0.2><0.5, 0.5> \\
& = \alpha <0.45, 0.1> \approx <0.818, 0.182> .
\end{align*}
$$
- On day 2, the umbrella appears, so $U_2 =true$. The prediction from $t=1$ to $t=2$ is
$$
\begin{align*}
P(R_2 | u_1) &= \sum_{r_1} P(R_2 | r_1)P(r_1 | u_1) \\
& = <0.7, 0.3> \times 0.818 + <0.3, 0.7> \times 0.182 \approx <0.627, 0.373>,
\end{align*}
$$
and updating it with the evidence for $t=2$ gives
$$
\begin{align*}
P(R_2 | u_1, u_2) &= \alpha P(u_2 |R_2)P(R_2 | u_1) = \alpha <0.9, 0.2><0.627, 0.373> \\
& = \alpha <0.565, 0.075> \approx <0.883, 0.117> .
\end{align*}
$$
Intuitively, the probability of rain increases from day 1 to day 2 because rain persists.

## Prediction
This is the task of computing the posterior distribution over the future state, given all evidence to date. That is, we wish to compute $P(X_{t+k} | e_{1:t})$ for some $k > 0$. In the umbrella example, this might mean computing the probability of rain three days from now, given all the observations to date. Prediction is useful for evaluating possible courses of action based on their expected outcomes.
The task of prediction can be seen simply as filtering without the addition of new evidence. In fact, the filtering process already incorporates a one-step prediction, and it is easy to derive the following recursive computation for predicting the state at $t + k + 1$ from a prediction for $t + k$:
$$
P(X_{t+k+1} | e_{1:t}) = \sum_{x_{t+k}} P(X_{t+k+1} | x_{t+k})P(x_{t+k} | e_{1:t}) .
$$
Naturally, this computation involves only the transition model and not the sensor model. It is interesting to consider what happens as we try to predict further and further into the future. It can be shown that the predicted distribution for rain converges to a fixed point $<0.5, 0.5>$, after which it remains constant for all time. This is the **stationary distribution** of the Markov process defined by the transition model.

## Smoothing
This is the task of computing the posterior distribution over a past state, given all evidence up to the present. That is, we wish to compute $P(X_k | e_{1:t})$ for some $k$ such that $0 \leq k < t$. In the umbrella example, it might mean computing the probability that it rained last Wednesday, given all the observations of the umbrella carrier made up to today. Smoothing provides a better estimate of the state than was available at the time, because it incorporates more evidence.
In anticipation of another recursive message-passing approach, we can split the computation into two parts—the evidence up to $k$ and the evidence from $k +1$ to $t$,
$$
\begin{align*}
P(X_k | e_{1:t}) &= P(X_k | e_{1:k}, e_{k+1:t}) \\
& = \alpha P(X_k | e_{1:k})P(e_{k+1:t} |X_k, e_{1:k}) \quad \text{(using Bayes’ rule)} \\
& = \alpha P(X_k | e_{1:k})P(e_{k+1:t} |X_k) \quad \text{(using conditional independence)} \\
& = \alpha f_{1:k} \times b_{k+1:t} .
\end{align*}
$$
where "$\times$" represents pointwise multiplication of vectors. Here we have defined a "backward" message $b_{k+1:t} =P(e_{k+1:t} |Xk)$, analogous to the forward message $f_{1:k}$. The forward message $f_{1:k}$ can be computed by filtering forward from 1 to $k$. It turns out that the backward message $b_{k+1:t}$ can be computed by a recursive process that runs backward from $t$:
$$
\begin{align*}
P(e_{k+1:t} |X_k) &= \sum_{x_{k+1}} P(e_{k+1:t} |X_k, x_{k+1})P(x_{k+1} |X_k) \quad  \text{(conditioning on Xk+1)} \\
& = \sum_{x_{k+1}} P(e_{k+1:t} | x_{k+1})P(x_{k+1} |X_k) \quad \text{(by conditional independence)} \\
& = \sum_{x_{k+1}} P(e_{k+1}, e_{k+2:t} | x_{k+1})P(x_{k+1} |X_k)
\\
& = \sum_{x_{k+1}} P(e_{k+1} | x_{k+1})P(e_{k+2:t} | x_{k+1})P(x_{k+1} |X_k),
\end{align*}
$$
where the last step follows by the conditional independence of $e_{k+1}$ and $e_{k+2:t}$, given $X_{k+1}$. Of the three factors in this summation, the first and third are obtained directly from the model, and the second is the “recursive call.” Using the message notation, we have
$$
b_{k+1:t} = \text{BACKWARD}(b_{k+2:t}, e_{k+1}) ,
$$
where BACKWARD implements the update described in previous equation. As with the forward recursion, the time and space needed for each update are constant and thus independent of $t$.

### An Example
Let us now apply this algorithm to the umbrella example, computing the smoothed estimate for the probability of rain at time $k=1$, given the umbrella observations on days 1 and 2. This is given by
$$
P(R_1 | u_1, u_2) = \alpha P(R_1 | u_1) P(u_2 |R_1) .
$$
The first term we already know to be $<0.818, 0.182>$, from the forward filtering process described earlier. The second term can be computed by applying the backward recursion:
$$
\begin{align*}
P(u_2 |R_1) &= \sum_{r_2} P(u_2 | r_2)P( | r_2)P(r_2 |R_1) \\
& = (0.9\times 1\times <0.7, 0.3>) + (0.2\times 1\times <0.3, 0.7>) = <0.69, 0.41> .
\end{align*}
$$
Using previous equation we find that the smoothed estimate for rain on day 1 is
$$
P(R_1 | u_1, u_2) = \alpha <0.818, 0.182>\times <0.69, 0.41> \approx <0.883, 0.117>.
$$
Thus, the smoothed estimate for rain on day 1 is higher than the filtered estimate (0.818) in this case. This is because the umbrella on day 2 makes it more likely to have rained on day 2; in turn, because rain tends to persist, that makes it more likely to have rained on day 1.


## Most likely explanation
Given a sequence of observations, we might wish to find the sequence of states that is most likely to have generated those observations.
### Recall: The Hidden Markov Model
A Markov chain is useful when we need to compute a probability for a sequence of observable events. In many cases, however, the events we are interested in are **hidden**: we don’t observe them directly.
A hidden Markov model (HMM) allows us to talk about both observed events Hidden Markov model (like words that we see in the input) and hidden events (like part-of-speech tags) that we think of as causal factors in our probabilistic model.

| ![HMM](./assets/hmm.jpg) | 
|:--:| 
| *A hidden Markov model for relating numbers of ice creams eaten (the **observations**) to the weather (H or C, the **hidden variables**).* |

Hidden Markov models should be characterized by **three fundamental problems**:

 1. **Likelihood**: Given an **HMM** $\lambda = (A,B)$ and an observation sequence $O$, determine the likelihood $P(O|\lambda)$.
 2. **Decoding**: Given an observation sequence $O$ and an **HMM** $\lambda = (A,B)$, discover the best hidden state sequence $Q.$
 3. **Learning**: Given an observation sequence $O$ and the set of states in the **HMM**, learn the HMM parameters $A$ and $B$.
 
### Likelihood Computation: The Forward Algorithm
The first problem is to compute the likelihood of a particular observation sequence. For example, given the ice-cream eating HMM, what is the probability of the sequence *3 1 3*? More formally:
***Computing Likelihood**: Given an HMM $\lambda = (A,B)$ and an observation sequence $O$, determine the likelihood $P(O|\lambda)$.*

Let’s start with a slightly simpler situation. Suppose we already knew the weather and wanted to predict how much ice cream Jason would eat. This is a useful part of many HMM tasks. For a given hidden state sequence (e.g., *hot hot cold*), we can easily compute the output likelihood of *3 1 3*.

Let’s see how. First, recall that for hidden Markov models, each hidden state produces only a single observation. Thus, the sequence of hidden states and the sequence of observations have the same length.
Given this one-to-one mapping and the Markov assumptions that the probability of a particular state depends only on the previous state, for a particular hidden state sequence $Q = q_0,q_1,q_2,...,q_T$ and an observation sequence $O = o_1,o_2,...,o_T$ , the likelihood of the observation sequence is:
$$
P(O|Q) = \prod_{i=1}^{T} P(o_i |q_i)
$$
The computation of the joint probability of our ice-cream observation *3 1 3* and one possible hidden state sequence *hot hot cold* is as follows:
$$
P(3\;1\;3,hot\;hot\;cold) = P(hot|start) \times P(hot|hot) \times P(cold|hot) \times P(3|hot) \times P(1|hot) \times P(3|cold)
$$
Now that we know how to compute the joint probability of the observations with a particular hidden state sequence, we can compute the total probability of the observations just by summing over all possible hidden state sequences:
$$
P(O) = \sum_{Q} P(O, Q) = \sum_{Q} P(O|Q)P(Q)
$$
For our particular case, we would sum over the eight 3-event sequences *cold cold cold*, *cold cold hot*, that is,
$$
P(3\;1\;3) = P(3\;1\;3, cold\;cold\;cold) +P(3\;1\;3, cold\;cold\;hot) +P(3\;1\;3,hot\;hot\;cold) +...
$$
For an HMM with $N$ hidden states and an observation sequence of $T$ observations, there are $N^T$ possible hidden sequences. For real tasks, where $N$ and $T$ are both large, $N^T$ is a very large number, so we cannot compute the total observation likelihood by computing a separate observation likelihood for each hidden state sequence and then summing them.
Instead of using such an extremely exponential algorithm, we use an efficient $O(N^2T)$ algorithm called the **forward algorithm**. The forward algorithm is a kind of **dynamic programming** algorithm, that is, an algorithm that uses a table to store intermediate values as it builds up the probability of the observation sequence. The forward algorithm computes the observation probability by summing over the probabilities of all possible hidden state paths that could generate the observation sequence, but it does so efficiently by implicitly folding each of these paths into a single forward trellis.

Each cell of the forward algorithm trellis $\alpha_t(j)$ represents the probability of being in state $j$ after seeing the first t observations, given the automaton $\lambda$. The value of each cell $\alpha_t(j)$ is computed by summing over the probabilities of every path that could lead us to this cell. Formally, each cell expresses the following probability:
$$
\alpha_t(j) = P(o_1,o_2 ...o_t ,q_t = j|\lambda)
$$
Here, $q_t = j$ means the $t^{th}$ state in the sequence of states is state $j$. We compute this probability $\alpha_t(j)$ by summing over the extensions of all the paths that lead to the current cell. For a given state $q_j$ at time $t$, the value $\alpha_t(j)$ is computed as
$$
\alpha_t(j) = \sum_{i = 1}^{N} \alpha_{t-1}(j)a_{i j}b_j(o_t)
$$
The three factors that are multiplied in this equation in extending the previous paths to compute the forward probability at time t are:

- $\alpha_{t-1}(j)$: the **previous forward path probability** from the previous time step
- $a_{ij}$: the **transition probability** from previous state $q_i$ to current state $q_j$
- $b_j(o_t)$: the **state observation likelihood** of the observation symbol $o_t$ given the current state $j$

Algorithm is done in three steps:
1. **Initialization:**
$$
\alpha_1(j) = \pi_jb_j(o_1)  \;\;1 \leq j \leq N
$$
2. **Recursion:**
$$
\alpha_t(j) = \sum_{i = 1}^{N} \alpha_{t-1}(j)a_{i j}b_j(o_t) \;\; 1 \leq j \leq N,1 < t \leq T
$$
3. **Termination:**
$$
P(O|\lambda) =\sum_{i=1}^{N} \alpha_T (i)
$$

The pseudocode of the forward algorithm:
``` java
function FORWARD(observations of len T, state-graph of len N) returns forward-prob
	create a probability matrix forward[N,T]
	for each state s from 1 to N do 				; initialization step
		forward[s,1]←pi(s) ∗ b_s(o_1)
	for each time step t from 2 to T do				; recursion step
		for each state s from 1 to N do
			forward[s,t] = sum(forward[j ,t-1] ∗ a_{j,s} ∗ b_s(o_t) for j=1 to N)
	forwardprob = sum(forward[s,T] for s=1 to N)	; termination step
	return forwardprob
```
### Decoding: The Viterbi Algorithm
For any model, such as an HMM, that contains hidden variables, the task of determining which sequence of variables is the underlying source of some sequence of observations is called the **decoding** task. In the ice-cream domain, given a sequence of ice-cream observations *3 1 3* and an HMM, the task of the decoder is to find the best hidden weather sequence (*H H H*). More formally,
***Decoding**: Given as input an HMM $\lambda = (A,B)$ and a sequence of observations $O = o_1,o_2,...,o_T$ , find the most probable sequence of states $Q = q_1q_2q_3 ...q_T$.*

The most common decoding algorithms for HMMs is the **Viterbi** algorithm. Like the forward algorithm, Viterbi is a kind of **dynamic programming** Viterbi algorithm that makes uses of a dynamic programming trellis.

The idea is to process the observation sequence left to right, filling out the trellis. Each cell of the trellis, $v_t(j)$, represents the probability that the HMM is in state $j$ after seeing the first $t$ observations and passing through the most probable state sequence $q_1,...,q_{t−1}$, given the automaton $\lambda$. The value of each cell $v_t(j)$ is computed by recursively taking the most probable path that could lead us to this cell. Formally, each cell expresses the probability
$$
v_t(j) = \max _{q_1,...,q_{t−1}} P(q_1...q_{t−1},o_1,o_2 ...o_t ,q_t = j|\lambda)
$$

Note that we represent the most probable path by taking the maximum over all possible previous state sequences. Like other dynamic programming algorithms, Viterbi fills each cell recursively. Given that we had already computed the probability of being in every state at time $t-1$, we compute the Viterbi probability by taking the most probable of the extensions of the paths that lead to the current cell. For a given state $q_j$ at time $t$, the value $v_t(j)$ is computed as
$$
v_t(j) = \max _{i=1} ^{N} v_{t−1}(i) a_{i j} b_j(o_t)
$$
The three factors that are multiplied in this equation for extending the previous paths to compute the Viterbi probability at time t are:

- $v_t(j)$: the **previous Viterbi path probability** from the previous time step
- $a_{i j}$: the **transition probability** from previous state $q-i$ to current state $q_j$
- $b_j(o_t)$: the **state observation likelihood** of the observation symbol $o_t$ given the current state $j$

The pseudocode of the viterbi algorithm:
``` java
function VITERBI(observations of len T,state-graph of len N) returns best-path, path-prob
	create a path probability matrix viterbi[N,T]
	for each state s from 1 to N do
		viterbi[s,1] = pi(s) * b_s(o_1)
		backpointer[s,1] = 0
	for each time step t from 2 to T do
		for each state s from 1 to N do
			viterbi[s,t] = max(viterbi[j,t-1] * a_{j,s} * b_s(o_t)) for j=1 to N
			backpointer[s,t] = argmax(viterbi[j,t-1] * a_{j,s} * b_s(o_t) for j=1 to N)
	bestpathprob = max(viterbi[s,T] for s=1 to N)
	bestpathpointer = argmax(viterbi[s,T] for s=1 to N)
	bestpath = the path starting at state bestpathpointer, that follows backpointer[] to states back in time
	return bestpath, bestpathprob
```
Note that the Viterbi algorithm is identical to the forward algorithm except that it takes the **max** over the previous path probabilities whereas the forward algorithm takes the **sum**.

### HMM Training: The Forward-Backward Algorithm
We turn to the third problem for HMMs: learning the parameters of an HMM, that is, the $A$ and $B$ matrices. Formally,
***Learning**: Given an observation sequence $O$ and the set of possible states in the HMM, learn the HMM parameters $A$ and $B$.*

The input to such a learning algorithm would be an unlabeled sequence of observations $O$ and a vocabulary of potential hidden states $Q$. Thus, for the ice cream task, we would start with a sequence of observations $O = \{1,3,2,...\}$ and the set of hidden states $H$ and $C$.
The standard algorithm for HMM training is the **forward-backward**, or **Baum-Welch** algorithm, a special case of the Expectation-Maximization or EM algorithm.
The algorithm will let us train both the transition probabilities $A$ and the emission probabilities $B$ of the HMM. EM is an iterative algorithm, computing an initial estimate for the probabilities, then using those estimates to computing a better estimate, and so on, iteratively improving the probabilities that it learns.

To understand the algorithm, we need to define a useful probability related to the forward probability and called the backward probability. The backward probability $\beta$ is the probability of seeing the observations from time $t+1$ to the end, given that we are in state $i$ at time $t$ (and given the automaton $\lambda$):
$$
\beta_t(i) = P(o_{t+1},o_{t+2} ...o_T |q_t = i,\lambda)
$$
It is computed inductively in a similar manner to the forward algorithm.

1. **Initialization:**
$$
\beta_T (i) = 1, \;\; 1 \leq i \leq N
$$
2. **Recursion:**
$$
\beta_t(i) =\sum_{j=1}^{N} a_{ij} b_j(o_t+1) \beta_{t+1}(j), \;\; 1 \leq i \leq N,1 \leq t < T
$$
3. **Termination:**
$$
P(O|\lambda) =\sum_{j=1}^{N} \pi_j b_j(o_1) β_1(j)
$$
Here is the pseudocode of this algorithm:
``` java
function FORWARD-BACKWARD(ev, prior) returns a vector of probability distributions
	inputs: ev, a vector of evidence values for steps 1,...,t
			prior, the prior distribution on the initial state, P(X0)
	local variables: fv, a vector of forward messages for steps 0,...,t
					 b, a representation of the backward message, initially all 1s
					 sv, a vector of smoothed estimates for steps 1,...,t
	fv[0] = prior
	for i = 1 to t do
		fv[i] = FORWARD(fv[i − 1], ev[i])
	for i = t downto 1 do
		sv[i] = NORMALIZE(fv[i] * b)
		b = BACKWARD(b, ev[i])
	return sv
```

## Resources
[1] Stuart Russell and Peter Norvig. Artificial Intelligence: A Modern Approach. 4th ed. Pearson Education, Inc
[2] Speech and Language Processing. Daniel Jurafsky & James H. Martin. https://web.stanford.edu/~jurafsky/slp3/A.pdf (Visited: 12/4/2021)

