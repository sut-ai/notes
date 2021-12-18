# **Bayesian Networks Representation**

## **Table of contents:**
- [Intoduction](#Intoduction)
	- [Probabilistic Models](#Probabilistic-Models)
- [Independence](#Independence)
	- [Conditional Independence](#Conditional-Independence)
	- [Chain Rule](#Chain-Rule)
- [Bayes' Nets](Bayes'-Nets)
	- [Problems with joint distribution tables](#Problems-with-joint-distribution-tables)
	- [Graphical Notation](#Graphical-Notation)
	- [Semantics](#Semantics)
	- [Probabilities in Bayes' Nets](#Probabilities-in-Bayes'-Nets)
	- [Causality in Bayes' Nets](#Causality-in-Bayes'-Nets)
	- [Space Efficiency](#Space-Efficiency)
- [Independence in Bayes' Nets](#Independence-in-Bayes'-Nets)
	- [Independency Assumptions](#Independency-Assumptions)
	- [D-separation](#D-separation)
		- [Causal Chain](#Causal-Chain)
		- [Common Cause](#Common-Cause)
		- [V-Structure](#V-Structure)
	- [Reachability](#Reachability)
	- [Active/Inactive Paths](#Active/Inactive-Paths)
	- [Structure Implications](#Structure-Implications)
- [Conclusion](#Conclusion)
- [References](#References)

## Intoduction

Imagine this scenario. You want to know if the food in your fridge is going to spoil. You can hear your fridge humming. Does that change the probability of the food spoiling? What if you see that the fridge door is open? Are these evidences even related to the food spoiling? In this lecture note, we want to introduce a probabilistic model to study this kind of problems. 

### Probabilistic Models

So, what is a probabilistic model? A model is a simplification of the real world; it describes how portions of the world work. Since models are simplifications, they do not contain all the details and may not account for every variable or interaction between variables. As George E. P. Box, a British statistician one said: "All models are wrong; but some are useful."

We use probabilistic models to reason about unknown variables, given evidences. This can be done in three ways:

- Diagnostic inference
	Means going from effects to causes, or in other words **explaining** the effects. For instance, given that the food spoiled, infer the probability of the fridge being broken.
- Causal inference
	Means going from causes to effects, or in other words **predicting** the effects. For example, given that the power is cut, find the probability of the food spoiling.
- Intercausal inference
	Means inferring probabilities between causes of a common effect, which is often called **explaining away**. For example, if we know that the food spoiled and the fridge door was open, the probability of the fridge being broken gets very low, although the door being open is independent of the fridge being broken.

## Independence

Two variables are *independent* if:

![formula1](/assets/T_Formula1.PNG)

If two variables are independent, their joint distribution is the product of their distributions. This can also be shown in another way:

![formula2](/assets/T_Formula2.PNG)

We show independency as:

![formula3](/assets/T_Formula3.PNG)

Note that real life joint distributions are at best close to independent. Independence is a *modelling assumption*. We'll get back to assumptions later.

### Conditional Independence

Unconditional (absolute) independence is very rare to come by, because the variables in a problem are usually all correlated. Conditional independence is our most basic and robust form of knowledge about uncertain environments. 

X is conditionally independent of Y given Z

![formula4](/assets/T_Formula4.PNG)

if and only if

![formula5](/assets/T_Formula5.PNG)

or equivalently, if and only if

![formula6](/assets/T_Formula6.PNG)

### Chain Rule

## Bayes' Nets

### Problems with joint distribution tables

### Graphical Notation

### Semantics

### Probabilities in Bayes' Nets

### Causality in Bayes' Nets

### Space Efficiency

## Independence in Bayes' Nets

### Independency Assumptions

### D-separation

#### Causal Chain

#### Common Cause

#### V-Structure

### Reachability

### Active/Inactive Paths

### Structure Implications

## Conclusion

## References