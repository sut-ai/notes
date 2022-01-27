# Introduction to Markov Decision Process

## Table of Contents

- [Introduction to Markov Decision Process](#introduction-to-markov-decision-process)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Markov Decision Process](#markov-decision-process)
    - [States](#states)
    - [Actions](#actions)
    - [Transition Model](#transition-model)
    - [Rewards](#rewards)
    - [Formalization](#formalization)
    - [Policy](#policy)
  - [MDP Search Trees](#mdp-search-trees)
  - [Utilities of Sequences](#utilities-of-sequences)
    - [Discounting](#discounting)
    - [Finite and infinite horizons](#finite-and-infinite-horizons)
      - [Infinite Utilities](#infinite-utilities)
      - [Stationary Preference](#stationary-preference)
  - [Bellman Equation](#bellman-equation)
  - [Time Limited Values](#time-limited-values)
  - [Implementation](#implementation)
  - [Real Life Example](#real-life-example)
  - [Conclusion](#conclusion)
  - [References](#references)

## Introduction

In nature, learning by trial and error is the most common way of learning. learning by trial and error is called Reinforcement Learning (RL) in computer literature. Markov Decision Process (MDP) is a foundational element of formulating RL mathematically. In a typical RL problem, there is a learner called agent, which makes decisions. The surrounding which agent interacts with is called __environment__. The environment provides rewards and a new state based on the actions the agent perform. All these could be modeled in a single MDP. The environment is modeled with __states__ and agent's possible decisions are modeled by __actions__. Since In real problems unexpected events may happen, agent's actions may not lead to expected results. This is why MDPs with __stochastic transition models__ are used. In RL, agent is not told how it should act but it is presented with rewards whether positive or negative based on its actions. These rewards are modeled by __reward function__ in an MDP. Solving an MDP means to find the best way that the agent can act. There are different ways to solve an MDP which all of them are based on __Bellman equation__, So in order to find the best guideline for the agent Bellman equation should be solved.

In this note, first MDP formalization is introduced. A way to represent MDP is MDP search trees.  MDP search trees are explained in the next part. At the end Bellman equation is discussed as the way of solving an MDP.

## Markov Decision Process

MDP is based on Markov chain. A Markov chain is a mathematical system that experiences transitions from one state to another according to certain probabilistic rules. The chain holds the following independence assumption.

$$ P(S_{t+1}|S_t, ..., S_0)=P(S_{t+1}|S_t) $$

$S_t$ represents the state at time $t$ . Intuitively, $S_t$ retains all of the information about the history that can affect the future states so *the future is conditionally independent of the past, given the present*.

### States

A representation of the environment's features at a specific time is called state and shown by $s$. Thus, any input from the agent's sensors can play an important role in state formation. The $S$ state set is a set of different states, represented as $s$ which shows the status of the environment. Each MDP consists of some states, which some of them are stopping state or so called terminal state. Terminal state is a state in which no action could be taken. When agent enters a stopping state, sum of rewards {which can be also called return} can be computed.

As an example, consider a robot car that wants to reach its destination as quickly as possible. The engine condition of this car can be cool, warm, or overheated depending on its speed.So in the above example, set of states would be {cool, warm, overheated}. Note that overheated is a terminal state.

<p align="center">
<image src="cool.jpg" width="30%">
<image src="warm.jpg" width="30%">
<image src="overheated.jpg" width="30%">
</p>

### Actions

Actions are set of operations an agent is allowed to do in the given environment. The set of actions is usually shown with $A$. In the robot car example, the machine can increase or decrease its speed. So the actions' set would be {slow, fast}

<p align="center">
<image src="action.jpg" width="60%">
</p>

### Transition Model

At each state, the agent decides which action to perform. The resulting state ($s'$) depends on both the current state ($s$) and the action performed by the agent ($a$). The transition model $T(s, a, s')$ gives probability $P(s'|s, a)$, that is, the probability of landing up in the new  state $s'$ given that the agent takes an action, $a$, in the given state, $s$.

Environments can be divided into two types according to their transition models.

- Determined environment: In a determined environment, if a certain action ($a$) is taken, the resulting state is the expected state ($s'$) with probability 1.
- Stochastic environment: In a stochastic environment, if the same action (a) is taken, with a certain probability ,e.g. $0.8$, the resulting state  will be the expected state and there is $0.2$ probability that the resulting state is not the expected state. Here, for the state $s$, the action $a$ and next state $s'$ the transition model is $T(s', a, s) = P(s'| s, a) = 0.8$.

In the machine robot example, we assume that the environment is stochastic. The probability of transitions is determined based on the current state, action, and next state.
One of the possible transitions is shown in the figure below.

<p align="center">
<image src="transition.jpg">
</p>

### Rewards

The reward function, quantifies the usefulness of taking a specific action and entering a specific state. These rewards incorporate the action costs in addition to any prizes or penalties that may be given. Negative rewards are called punishments. The general form of the reward function is shown as $R(s, a, s')$. But in some MDPs the reward function is independent from the resulting state or both resulting state and action. In those models reward function is shown with $R(s, a)$ and $R(s)$ respectively.

For a particular environment, the domain knowledge plays an important role in the assignment of rewards for different states as minor changes in the reward do matter for finding the optimal solution to an MDP problem.
Sum of the rewards is called *utility* or *return*.

Usually the performance of the agent, which is also called *utility*, is measured by the sum of rewards on the visited path. But other utility functions are also possible. General form of utility is shown by $U_h[s_0,s_1, s_2, ..., s_N]$.  

In the robot car example, the robot must reach its destination as quickly as possible, so the high-speed reward must be considered more than the low-speed, e.g. +2 and +1 may be considered. But reward should be assigned in a way to prevent the robot from going to overheated state. So -10 is assigned to entering the overheated state.

<p align="center">
<image src="reward.jpg">
</p>

### Formalization

By combining the concepts which were explained above the complete formulation for MDP is achieved as below.

- A set of possible states $S$.
- A set of possible actions $A$.
- A transition function $T(s, a, s')$: Probability that taking action $a$ in state $s$ leads to $s'$, i.e., $P(s'| s, a)$
- Also called the model or the dynamic.
- A reward function $R(s, a, s')$: Sometimes reward only depends on the resulting state or on the resulting state and action. So $R(s, a, s')$ can be replaced by $R(s')$ or $R(s', a)$
- A start state
- Maybe a terminal state

### Policy

The policy is a function that takes the state as an input and outputs the action to be taken. Policy $\pi : S \rightarrow A$ is a set of commands that the agent follows.

The policy which maximizes the expected utility among all the possible policies is called optimal policy and shown with $\pi^\star$.

Finding $\pi^\star$ is usually referred to as solving the MDP. There are different algorithms which attempt to find $\pi^\star$ such as value iteration and policy evaluation.

## MDP Search Trees

In expectimax tree the path which leads to the most reward is desired. MDP problems could be demonstrated as a search tree. But there are two main differences.

- In MDP Rewards are assigned to edges rather than nodes.
- In MDP utility is the sum of rewards gained in the path to the terminal state. But in expectimax utility is the the value assigned to the terminal state.

In the robot car example, the search is as shown below.

<p align="center">
<image src="tree.png">
</p>

## Utilities of Sequences

It's easy to choose between sequence of [1, 2, 2] and [2, 3, 4]. Obviously more utility as well as more reward in each step is gained . But choosing between [1, 0, 0] and [0, 0, 1] is not that easy. This motivates us to consider the time which the reward is being received. In most cases, it's preferred to gain the rewards as soon as possible.

### Discounting

Consider gaining reward $r_i$ in time step $t_i$. In discounting, $\sum \gamma^t r_i$ is considered instead of $\sum r_i$ as utility function such that $0\lt \gamma \lt 1$. $\gamma=1$ is just equivalent to simple summation. Apart from what we have said so far, this method has other advantages, which will be mentioned below briefly.

### Finite and infinite horizons

Decision making problems may be of two types. Some may be __finite horizon__ and some __infinite horizon__. Finite horizon means that there is a fixed time N after which nothing matters. In other words what happens after the fixed time is not analysed. To be mathematically shown, $U_h([s_0, s_1,...,s_{N+k}]) = U_h([s_0, s_1,..., s_N ])$
for all $k > 0$ . In finite horizon problems, the optimal action in a given state could change over time. That is, when there is little time left, the agent should take risk to gain reward before the deadline is reached. Because of this change in optimal action in a given state over time, the optimal policy for a finite horizon is non stationary. However, with no fixed time limit, there is no reason to behave differently in the same state at different times. Hence, the optimal policy depends only on the current state, and the optimal policy is stationary. Note that infinite horizon does not necessarily mean that all state sequences are infinite, it just means that there is no fixed deadline.

#### Infinite Utilities

In infinite horizon problems, calculating an upper bound on the gained utility is desired to distinguish different policies appropriately. But defining utility as $\sum r_i$ does not satisfy the mentioned condition. But if discounting is applied, according to the geometric sequence sum $\sum \gamma^t r_i < r_{max} \sum \gamma^t = \frac{r_{max}}{1 - \gamma}$ .

#### Stationary Preference

In infinite horizon only if utility is defined as $\sum r_i$ or $\sum \gamma^t r_i$ the agent's preferences between state sequences are stationary. Stationarity for preferences means

$[s, s_0, s_1, s_2,  ... ] > [s, s_0', s_1', s_2',  ... ] \iff [s_0, s_1, s_2,  ... ] > [s_0', s_1', s_2',  ... ].$

This means that if a sequence is preferred to another, it is also preferred after one state is passed. Explaining in simple words, this means that if one future is preferred  to another starting tomorrow, then it is still preferred if it were to start today instead.

## Bellman Equation

Due to previous discussions, now objective is to find policy that maximizes expected discounted utility.
$$ \max_{\pi} E \Bigg[ \sum_{i=0}^\infty \gamma^ir_i \Bigg] $$

Consider $V^\star(s)$ as expected utility starting from state $s$ and act according to $\pi^\star$, the optimal policy.

$Q^\star(s, a)$ is the expected utility starting from state $s$ then performing action $a$ and after that acting according to $\pi^\star$.

By considering all forward steps and choosing the step which maximizes the utility, $V^\star$ can be formulated as
$$ V^\star(s) = \max_{a} Q^\star(s, a) . $$
Then by calculating expectation on earned utility by marginalizing on all possible next states which can be succeeded after action $a$ in state $s$, $Q^\star$ will be written as
$$ Q^\star(s, a) = \sum_{s'}T(s, a, s')\big[R(s, a, s') + \gamma V^\star(s') \big].$$

Since $V^\star$ and $Q^\star$ are jointly recursive, It is possible to replace $Q^\star$ with $V^\star$.
$$ V^\star(s) = \max_{a} \sum_{s'}T(s, a, s')\big[R(s, a, s') + \gamma V^\star(s') \big] $$

Equation above is called as __bellman equation__.

By solving bellman equation for all states, optimal policy will be obtained.

## Time Limited Values

Solving bellman equation via tree has two major problems:

1. There are repeating sub problems, so there are a lots of overheads on the system.
2. The depth of the tree could be infinite.

An approach to solve this problem is to try solving the problem in $k$ steps which converge to final answer instead of one-shot solving.

Instead of $V^\star(s)$, Consider $V_k(s)$ which is the expected utility starting from state $s$ and acting according to $\pi^\star$ with consideration that the process will be terminated after $k$ steps. The idea is that with the growth of $k$, $V_k$ will converge to $V^\star$ so an approximate solution to the equations system can be obtained.

Equations below are the limited form of what was discussed about $V^\star$ and $Q^\star$ above.
$$ V_{k+1}(s) = \max_{a} Q_k(s, a) $$
$$ Q_k(s, a) = \sum_{s'}T(s, a, s')\big[R(s, a, s') + \gamma V_k(s') \big]$$

So $V_{k+1}$ can be calculated by
$$ V_{k+1}(s) = \max_{a} \sum_{s'}T(s, a, s')\big[R(s, a, s') + \gamma V_k(s') \big] .$$

For each state $s$, $V_{k+1}(s)$ will be calculated by iteration over all states and all actions, which has time complexity $O(|S||A|)$. So the total time complexity for each iteration will  be $O(|S|^2|A|)$.

## Implementation

States, terminal states, actions, transition and reward function should be implemented in order to define the MDP.
The robot car example explained above will be encoded as bellow

```Python
car_states = ['Cool', 'Warm', 'Over']
car_terminals = ['Over']
car_actions = ['fast', 'slow']


def car_transition(state, action, next_state):
    if state == 'Cool':
        if action == 'slow':
            if next_state == 'Cool':
                return 1.0
        if action == 'fast':
            if next_state == 'Cool':
                return 0.5
            if next_state == 'Warm':
                return 0.5
    if state == 'Warm':
        if action == 'slow':
            if next_state == 'Cool':
                return 0.5
            if next_state == 'Warm':
                return 0.5
        if action == 'fast':
            if next_state == 'Over':
                return 1.0
    return 0


def car_reward(state, action, next_state):
    if state == 'Cool':
        if action == 'slow':
            if next_state == 'Cool':
                return 1
            if next_state == 'Warm':
                return 0  # Impossible
            if next_state == 'Over':
                return 0  # Impossible
        if action == 'fast':
            if next_state == 'Cool':
                return 2
            if next_state == 'Warm':
                return 2
            if next_state == 'Over':
                return 0  # Impossible
    if state == 'Warm':
        if action == 'slow':
            if next_state == 'Cool':
                return 1
            if next_state == 'Warm':
                return 1
            if next_state == 'Over':
                return 0  # Impossible
        if action == 'fast':
            if next_state == 'Cool':
                return 0  # Impossible
            if next_state == 'Warm':
                return 0  # Impossible
            if next_state == 'Over':
                return -10
    if state == 'Over':
        if action == 'slow':
            if next_state == 'Cool':
                return 0  # Impossible
            if next_state == 'Warm':
                return 0  # Impossible
            if next_state == 'Over':
                return 0  # Impossible
        if action == 'fast':
            if next_state == 'Cool':
                return 0  # Impossible
            if next_state == 'Warm':
                return 0  # Impossible
            if next_state == 'Over':
                return 0  # Impossible
```

If we ignore the impossible cases, we will find that in this problem the reward is only a function of the action and next state. So we can rewrite reward function as below.

```Python
def car_reward(state, action, next_state):
    if action == 'slow':
        return 1
    if action == 'fast':
        if next_state == 'Over':
            return -10
        return 2
```

In order to find the optimal policy, while finding new values using value iteration, the action which maximizes the value should also be stored.

```Python
from typing import List
from typing import Tuple


def argmax(l: List) -> int:
    index_max = 0
    for i in range(len(l)):
        if l[i] > l[index_max]:
            index_max = i
    return index_max


def mdp_iterate(transition_function, reward_function, gamma: float, states: List, terminals: List, actions: List, v: List):
    new_v = []
    best_actions = []
    for i in range(len(states)):
        state = states[i]
        if state in terminals:
            new_v.append(v[i])
            best_actions.append(None)
            continue
        values_actions = []
        for action in actions:
            values_actions.append(sum([transition_function(state, action, states[j]) * (reward_function(state, action, states[j]) + gamma * v[j]) for j in range(len(states))]))
        new_v.append(max(values_actions))
        best_actions.append(actions[argmax(values_actions)])
    return new_v, best_actions


def mdp_solve(transition_function, reward_function, gamma: float, states: List, terminals: List, actions: List, iter: int):
    v = [0 for _ in range(len(states))]
    p = [None for _ in range(len(states))]
    for _ in range(iter):
        v, p = mdp_iterate(transition_function, reward_function, gamma, states, terminals, actions, v)
    return {states[i]: p[i] for i in range(len(states))}
```

By running the script below, this result will be generated which is the optimal policy.

```Python
>>> mdp_solve(car_transition, car_reward, 0.9, car_states, car_terminals, car_actions, 10)
{'Cool': 'fast', 'Warm': 'slow', 'Over': None}
```

## Real Life Example

In a TV quiz show, there are several levels. At each level, if the participant answers the question correctly, they will receive some prize. If the participant's answer is wrong, they leave the competition empty-handed. Before each stage begins, the participant can decide whether to continue or withdraw and leave the game with the reward which they already earned.

Beside states representing each level, There are three terminal states of *Win*, *Lost* and *Quit* in the game. Actions in each state are quit and play. The player will go to the quit state with probability of 1 if they decide to take action quit. otherwise they will pass the level i by probability of `win_ratio[i]` and go to the state which represents the next level.

So the play / quit decision problem can be modeled as an MDP as below.

<p align="center">
<image src="quiz.png">
</p>

Considering $100, $200, $300, $400 and $500 as rewards and 0.9, 0.7, 0.6, 0.3, 0.1 as win ratio for levels 0 to 4 respectively, The model can be implemented as below.

```Python
quiz_levels = [f'{i}' for i in range(5)]
quiz_terminals = ['Win', 'Lost', 'Quit']
quiz_states = quiz_levels + quiz_terminals
quiz_actions = ['play', 'quit']

quiz_win_ratio = [0.9, 0.7, 0.6, 0.3, 0.1]
quiz_win_amount = [100, 200, 300, 400, 500]


def quiz_transition(state, action, next_state):
    if state in quiz_terminals:
        return 0
    else:  # quiz levels
        state_level = int(state)
        if action == 'quit':
            if next_state == 'Quit':
                return 1
            else:
                return 0
        else:  # play
            if next_state == 'Win':
                if state_level == 4:
                    return quiz_win_ratio[4]
                else:
                    return 0
            elif next_state == 'Lost':
                return 1 - quiz_win_ratio[state_level]
            elif next_state == 'Quit':
                return 0
            else:
                next_state_level = int(next_state)
                if next_state_level == state_level + 1:
                    return quiz_win_ratio[state_level]
                else:
                    return 0


def quiz_reward(state, action, next_state):
    state_level = int(state)
    if action == 'quit':
        return 0
    else:  # play
        if next_state == 'Win':
            return quiz_win_amount[4]
        elif next_state == 'Lost':
            return -1 * sum(quiz_win_amount[:state_level])
        elif next_state == 'Quit':
            return 0
        else:
            return quiz_win_amount[state_level]
```

By running the script below, this result will be generated which is the optimal policy. Consider that in this specific problem, we don't want to discount the reward, So we set gamma to one.

```Python
>>> mdp_solve(quiz_transition, quiz_reward, 1, quiz_states, quiz_terminals, quiz_actions, 1000)
{'0': 'play',
 '1': 'play',
 '2': 'play',
 '3': 'quit',
 '4': 'quit',
 'Win': None,
 'Lost': None,
 'Quit': None}
```

In oreder to check the result acheived by the script above, expectation of reward earned in each step, is calculated. It can be seen that in level 0, 1, and 2 the expection is greater than zero so continuing the game is the optimal action in these states, but in level 3 and 4, the expectation is less than zero so it's better to quit the game.

$
E_0 = 0.9 \times 100 + 0.1 \times 0 = 90 \\
E_1 = 0.7 \times 200 + 0.3 \times -100 = 110 \\
E_2 = 0.6 \times 300 + 0.4 \times -300 = 60 \\
E_3 = 0.3 \times 400 + 0.7 \times -600 = -300 \\
E_4 = 0.1 \times 500 + 0.9 \times -1000 = -850
$
## Conclusion

In conclusion MDP is an appropriate tool to represent RL problems. In order to represent a problem using MDP, states, actions, transition model and rewards should be determined properly. Afterwards finding the optimal policy, i.e. the one with the most utility, is desired. In order to find the optimal utility, bellman equation should be solved. Algorithms such as value iteration and policy iteration attempt to solve the bellman equation feasibly. In the end take into consideration that in real world RL problems, the transition model and reward function are not known and must be learned.

## References

- Russell, Stuart, and Peter Norvig. "Artificial intelligence: a modern approach." (2002).
- <https://inst.eecs.berkeley.edu/~cs188/fa18/assets/notes/n4.pdf>
- <https://towardsdatascience.com/real-world-applications-of-markov-decision-process-mdp-a39685546026>
- <https://web.mit.edu/6.246/www/notes/L3-notes.pdf>
- <https://hub.packtpub.com/reinforcement-learning-mdp-markov-decision-process-tutorial/>
- <http://artint.info/html/ArtInt_160.html#stationary-Markov-Chain>
- <https://towardsdatascience.com/introduction-to-reinforcement-learning-markov-decision-process-44c533ebf8da>
- <https://www.geeksforgeeks.org/markov-decision-process/>
