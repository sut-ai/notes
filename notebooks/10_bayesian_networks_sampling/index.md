<div align="center">
    <br>
    <br>
    <br>
    <br>
    <br>
    <br>
    <br>
    <br>
    <h1 style="font-size: 40px; margin: 10px 0;">AI - Inference in Bayesian Networks: Sampling</h1>
    <h1 style="font-size: 20px; font-weight: 400;">Sharif University of Technology - Computer Engineering Department</h1>
    <br>
    <h4 style="font-size: 18px; font-weight: 400; color:#555">Alireza Honarvar, Navid Eslami, Ali Najibi</h4>
    <br>
    <br>
    <br>
    <br>
    <br>
</div>
<hr>

Table of Contents
==============

- [Introduction](#Introduction)
- [Basic Idea](#Basic-Idea)
- [Prior Sampling](#Prior-Sampling)
- [Rejection Sampling](#Rejection-Sampling)
- [Likelihood Weighting](#Likelihood-Weighting)
- [Gibbs Sampling](#Gibbs-Sampling)

# Introduction

In the past lecture note, it was shown that Inference
in Bayesian Networks, in general, is an intractable
problem. The natural approach now would be to try
and approximate the posterior probability. There are
several approximation methods for this problem, of
which we will discuss the ones based on randomized
sampling.

# Basic Idea

To compute an approximate posterior probability, one 
approach is to simulate the Bayes' Net's joint 
distribution. This can be achieved by drawing many samples 
from the joint distribution. Using these samples, we can 
approximate the probability of certain events.

Sampling has two main advantages:

- Learning: By getting samples from an unknown 
distribution, we can learn the associated probabilities.
- Performance: Getting a sample is much faster than 
computing the right answer.

The primitive element in any sampling algorithm is the 
generation of samples from a known
probability distribution. So the step-by-step algorithm is 
described in the following section.

# Sampling from Given Distribution
<center>

|   C   	| P(C) 	|
|:-----:	|:----:	|
|  red  	|  0.6 	|
| green 	|  0.1 	|
|  blue 	|  0.3 	|

</center>

# Prior Sampling

# Rejection Sampling

# Likelihood Weighting

# Gibbs Sampling
