# Inference in Bayes Nets 1

## Table of Contents
- [Introduction](#introduction)
- [Inference by Enumeration](#inference-by-enumeration)
    - [Algorithm Explanation](#algorithm-explanation)
    - [Algorithm Steps](#algorithm-steps)
    - [Algorithm Example](#algorithm-example)
- [Conclusions](#conclusions)
- [References](#references)

## Introduction
The basic task of a Bayesian network is to compute the posterior probability distributions for a set of query variables, given an observation of a set of evidence variables. This process is known as inference, but is also called Bayesian updating, belief updating or reasoning. This process is known as inference, but is also called Bayesian updating, belief updating or reasoning. There are two ways to approach this, either exact or approximate. Both approaches are worst-case NP-hard. An exact method obviously gives an exact result, while an approximate method tries to approach the correct outcome as close as possible. In this lecture, we discuss the exact inference method. Approximate (Sampling) method will be discussed in the next lecture. 

## Inference by Enumeration
- exact 
- expnential complexity

### Algorithm Explanation

The enumeration algorithm is a simple, brute-force algorithm for computing the distribution of a variable in a Bayes net. In this algorithm we partition all Bayes net variables into three groups:

1. evidence variables
1. hidden variables
1. query variables

This algorithm takes query variables and evidence variables as input, and outputs the distribution of query variables. 

The evidence e is whatever values you already know about the variables in the Bayes net. Evidence simplifies your work because instead of having to consider those variables’ whole distributions, you can assign them particular values, so they are no longer variables, they are constants. In the most general case, there is no evidence.

This algorithm has to compute a distribution over $X$, which, because $X$ is a discrete variable, means computing the probability that $X$ takes on each of its possible values (the values in its domain). The algorithm does this simply by looping through all of the possible values, and computing the probability for each one. Possible values are the result of marginalizing out hidden variables from entries which are consistent with given evidence. Note that if there is no evidence, then it is literally just computing the probabilities $P(X=x_i)$ for each xi in $X$’s domain. If there is evidence, then it is computing $P(e, X=xi)$ for each xi in $X$’s domain – that is, it is computing the probability that $X$ has the given value $x_i$ and the evidence is true – so in that case, we use the law of conditional probability, which says that $P(X=x_i | e) = \frac{P(e, X=x_i)}{P(e)}$. Once we have computed $P(e, X=x_i)$ for all $x_i$, we can just normalize those values to get the correct distribution $P(X | e)$.

### Algorithm Steps
We can summurize the explained algorithm in the following steps:

1. Select the entries consistent with the evidence.
2. Sum out the hidden variables to get joint distribution of query and evidence variables.
3. Normalize the distribution to get the distribution of query variables.

### Algorithm Example




## Variable Elimination
- exact
- worst case exponential complexity, often better

## 




## Conclusions

## References
- [Visualizing Inference in Bayesian Networks](http://www.kbs.twi.tudelft.nl/Publications/MSc/2006-JRKoiter-Msc.html)
- [Exact Inference in Bayes Nets](http://courses.csail.mit.edu/6.034s/handouts/spring12/bayesnets-pseudocode.pdf)