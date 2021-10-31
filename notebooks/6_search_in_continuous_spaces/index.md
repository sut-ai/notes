Search in Continuous Space
#
# **Table of Contents**
` `**TOC \o "1-3" \h \z \u [Introduction	 PAGEREF _Toc86421092 \h 2**](#_Toc86421092)**

[**Type of optimization techniques	 PAGEREF _Toc86421093 \h 2**](#_Toc86421093)

[*Linear programming (LP) and Nonlinear programming (NLP)	 PAGEREF _Toc86421094 \h 2*](#_Toc86421094)

[*Continuous optimization and Integer programming (IP)	 PAGEREF _Toc86421095 \h 2*](#_Toc86421095)

[*Constrained optimization and Unconstrained optimization	 PAGEREF _Toc86421096 \h 2*](#_Toc86421096)

[*Differentiable optimization and Non-differentiable optimization	 PAGEREF _Toc86421097 \h 3*](#_Toc86421097)

[Cost functions	 PAGEREF _Toc86421098 \h 3](#_Toc86421098)

[**Convexity	 PAGEREF _Toc86421099 \h 4**](#_Toc86421099)

[*Convex versus non-Convex Optimization Problems	 PAGEREF _Toc86421100 \h 5*](#_Toc86421100)

[*Local and global optimization:	 PAGEREF _Toc86421101 \h 5*](#_Toc86421101)

[Local optimization	 PAGEREF _Toc86421102 \h 5](#_Toc86421102)

[Global optimization	 PAGEREF _Toc86421103 \h 6](#_Toc86421103)

[*Locally optimal for a convex optimization problem	 PAGEREF _Toc86421104 \h 6*](#_Toc86421104)

[*Hessian being positive semi-definite iff convexity (Theorem one: One dimensional)	 PAGEREF _Toc86421105 \h 6*](#_Toc86421105)

[*Hessian being positive semi-definite iff convexity (Theorem two: Multi-dimensional)	 PAGEREF _Toc86421106 \h 7*](#_Toc86421106)

[**Gradient	 PAGEREF _Toc86421107 \h 8**](#_Toc86421107)

[*Gradient Descent	 PAGEREF _Toc86421108 \h 8*](#_Toc86421108)

[*Challenges with gradient descent	 PAGEREF _Toc86421109 \h 10*](#_Toc86421109)

[Local minima and saddle points	 PAGEREF _Toc86421110 \h 10](#_Toc86421110)

[Vanishing and Exploding Gradients	 PAGEREF _Toc86421111 \h 10](#_Toc86421111)

[**Conclusion	 PAGEREF _Toc86421112 \h 11**](#_Toc86421112)

[**References	 PAGEREF _Toc86421113 \h 11**](#_Toc86421113)


#
#


# Introduction
In real life optimization problems, we mostly face continuous environments which are made of continuous variables such as time and location. Thus, in this lecture note we will discuss some useful search methods in continuous spaces. The typical problems are linear programming (LP) and diverse kinds of nonlinear programming (such as convex programming, quadratic programming or semidefinite programming). In this lecture note we will discuss different types of optimization techniques in continuous spaces, the differences between different types of optimization techniques, how convexity will help us solve optimization problems easier, difference between local and global optimization, gradient descent method and the challenges with finding the best learning rate.

# Type of optimization techniques
An essential step to optimization technique is to categorize the optimization model since the algorithms used for solving optimization problems are customized as per the nature of the problem. Various optimization problem types are: 
## Linear programming (LP) and Nonlinear programming (NLP)
Linear programming is a method to achieve the best outcome in a mathematical model whose requirements are represented by linear relationships whereas nonlinear programming is a process of solving an optimization problem where the constraints or the objective functions are nonlinear. Thus, this is the main difference between linear and nonlinear programming.
## Continuous optimization and Integer programming (IP)
In order to conceptualize the difference, linear programming can solve problems about minimizing (or maximizing) an objective function by continuous variables. For instance, maybe the optimal solution for a problem to be x1=5,46 and x2=2,65. But in integer programming we can only use integers as variables and we can’t have fractional numbers.
## Constrained optimization and Unconstrained optimization 
**Constrained optimization problems** consider the problem of optimizing an objective function subject to constraints on the variables. In general terms,

minimize fx

subject to    cix=0    ∀i ϵ ε

cix≤0    ∀i ϵ τ 

Where *f and* the functions ci(x) are all smooth, real-valued functions on a subset of Rn and ε and τ are index sets for equality and inequality constraints, respectively. The feasible set is the set of points x that satisfy the constraints.

We denote the set of points for which all the constraints are satisfied as C, and say that any x ∈ C (resp. x ∈/ C) is feasible (resp. infeasible)

***Unconstrained optimization problems*** the answers are constrained into being subject of set C as the picture bellow shows: 

![](Aspose.Words.f2571b5b-5e5e-42a8-b357-02f03b593c59.001.png)





*Figure  SEQ Figure \\* ARABIC 1. constrained vs unconstrained optimization*
![](Aspose.Words.f2571b5b-5e5e-42a8-b357-02f03b593c59.002.png)

## Differentiable optimization and Non-differentiable optimization
*Figure  SEQ Figure \\* ARABIC 2. Non-differentiable function*
![](Aspose.Words.f2571b5b-5e5e-42a8-b357-02f03b593c59.003.png)Non-differentiable optimization is a category of optimization that deals with objective that for a variety of reasons is non-differentiable and thus non-convex. The functions in this class of optimization are generally non-smooth. These functions although continuous often contain sharp points or corners that do not allow for the solution of a tangent and are thus non-differentiable. In practice non-differentiable optimization encompasses a large variety of problems and a single one-size fits all solution is not applicable however solution is often reached through implementation of the sub gradient method. Non-differentiable functions often arise in real world applications and commonly in the field of economics where cost functions often include sharp points.

![](Aspose.Words.f2571b5b-5e5e-42a8-b357-02f03b593c59.004.png)
### Cost functions
In many cases, particularly economics the cost function which is the objective function of an optimization problem is non-differentiable. These non-smooth cost functions may include discontinuities and discontinuous gradients and are often seen in discontinuous physical processes. Optimal solution of these cost functions is a matter of importance to economists but presents a variety of issues when using numerical methods thus leading to the need for special solution methods.

In this lecture we don’t discuss non-differential optimization and non-smooth functions and the text above was for introduction and further information on this topic.

# Convexity
A set C is convex if for any two points x,y ∈C ,αx+1-αy∈C for all α∈0,1, i.e., all points on the line between x and y also lie in C. A function f is convex if its domain C is convex and

fαx+1-αy≤αfx+1-αfy (\*)

*Figure  SEQ Figure \\* ARABIC 3. In convex function f, for every two point x,y∈domainf, the line segment between them lies above the graph of f.*
![](Aspose.Words.f2571b5b-5e5e-42a8-b357-02f03b593c59.005.png)![](Aspose.Words.f2571b5b-5e5e-42a8-b357-02f03b593c59.006.jpeg)for all α∈0,1, i.e., the line segment betweenx,fxandy,fy, which is the chord from x to y, lies above the graph of f.


For an affine function we always have equality in (\*), so all affine (and therefore also linear) functions are both convex and concave. Conversely, any function that is convex and concave is affine.

A function is convex if and only if it is convex when restricted to any line that intersects its domain. In other words, f is convex if and only if for all x ∈ domain (f) and all v, the function gt=fx+tv is convex (on its domain, {t | x + t v ∈ domain (f) }). This property is very useful, since it allows us to check whether a function is convex by restricting it to a line.

*Figure  SEQ Figure \\* ARABIC 4. A Non-convex function vs a Convex function*

![](Aspose.Words.f2571b5b-5e5e-42a8-b357-02f03b593c59.007.png)![](Aspose.Words.f2571b5b-5e5e-42a8-b357-02f03b593c59.008.png)
## Convex versus non-Convex Optimization Problems
*Figure  SEQ Figure \\* ARABIC 5. convex region*
![](Aspose.Words.f2571b5b-5e5e-42a8-b357-02f03b593c59.009.png)![](Aspose.Words.f2571b5b-5e5e-42a8-b357-02f03b593c59.010.png)A *convex optimization problem* is a problem where all of the constraints are convex functions, and the objective is a convex function if minimizing, or a concave function if maximizing. In a convex optimization problem, the feasible region is a convex region, as pictured below.

With a convex objective and a convex feasible region, there can be only one optimal solution, which is globally optimal.  Several methods will either find the globally optimal solution, or prove that there is no feasible solution to the problem.  Convex problems can be solved efficiently up to very large size.

*Figure  SEQ Figure \\* ARABIC 6. Non-convex region*

![](Aspose.Words.f2571b5b-5e5e-42a8-b357-02f03b593c59.011.png)![](Aspose.Words.f2571b5b-5e5e-42a8-b357-02f03b593c59.012.png)A *non-convex optimization problem* is any problem where the objective or any of the constraints are non-convex, as pictured below.

Such a problem may have multiple feasible regions and multiple locally optimal points within each region.  It can take time exponential in the number of variables and constraints to determine that a non-convex problem is infeasible, that the objective function is unbounded, or that an optimal solution is the "global optimum" across all feasible regions.
##
## Local and global optimization:
### Local optimization 
In local optimization, the compromise is to give up seeking the optimal x, which minimizes the objective over all feasible points. Instead we seek a point that is only locally optimal, which means that it minimizes the objective function among feasible points that are near it, but is not guaranteed to have a lower objective value than all other feasible points. 

Local optimization methods can be fast, can handle large-scale problems, and are widely applicable, since they only require differentiability of the objective and constraint functions.

There are several disadvantages of local optimization methods, beyond (possibly) not finding the true, globally optimal solution. The methods require an initial guess for the optimization variable. This initial guess or starting point is critical, and can greatly affect the objective value of the local solution obtained.
### Global optimization
In global optimization, the true global solution of the optimization problem is found; the compromise is efficiency. The worst-case complexity of global optimization methods grows exponentially with the problem size. 

The hope is that in practice, for the particular problem instances encountered, the method is far faster. While this favorable situation does occur, it is not typical. Even small problems, with a few tens of variables, can take a very long time (e.g., hours or days) to solve. Global optimization is used for problems with a small number of variables, where computing time is not critical, and the value of finding the true global solution is very high.
##
## Locally optimal for a convex optimization problem 
A fundamental property of convex optimization problems is that any locally optimal point is also (globally) optimal. To see this, suppose that x is locally optimal for a convex optimization problem, i.e., x is feasible and

f\_0(x) = inf{f\_0(z) | z   feasible, kz - xk^2 ≤ R} , (@)

for some R>0. Now suppose that x is not globally optimal, i.e., there is a feasible y such that  f0y<f0x. Evidently y–x2>R, since otherwise f0y<f0x.Consider the point z given by 

z=1-θx+θy,θ=R2y–x2

` `Then we have z–x=R/2<R, and by convexity of the feasible set, z is feasible. By convexity of f0 we have 

f0z≤1-θf0x+θf0y<f0x,

which contradicts (@). Hence there exists no feasible y with f0y<f0x , i.e., x is globally optimal.

## Hessian being positive semi-definite iff convexity (Theorem one: One dimensional) 
One way of checking a function’s convexity is by checking its Hessian matrix. let ∅ ≠M ⊆ Rn be a convex set and let fx be a function differentiable on an open superset of M then fx is convex on M if and only if for every x1,x2 ∈M

fx2-fx1≥∇f(x1)T x2- x1 (\*)

Proof: ⇒let x1, x2 ∈M and λ ∈0,1 be arbitrary . Then 

f1-λx1+ λx2≤1-λfx1+λfx2

f1-λx1+ λx2-fx1≤ λfx2-fx1

f1-λx1+ λx2-fx1λ≤fx2-fx1 

By the limit transition λ ⟶0 we get (\*) utilizing the chain rule for the derivative of a composite function gλ=f(x1+ λx2-x1 with respect to λ.

⇐ Let x1, x2 ∈M and consider a convex combination x= λ1x1+ λ2x2 by (\*) we have:

fx1-fx≥∇fxT x1- x= λ2∇fxTx1-x2

fx2-fx≥∇fxT x2- x= λ2∇fxTx2-x1

Multiply the first inequality by λ1 , the one by λ2 and summing up we get 

λ1fx1-fx+ λ2 (fx2-fx)≥0 

or λ1fx1+λ2 fx2≥fx

## Hessian being positive semi-definite iff convexity (Theorem two: Multi-dimensional) 
Now we prove the same property for n-dimensional functions. let ∅ ≠M ⊆ Rn be an open convex set of dimension n, and suppose that a function f : M → R is twice continuously differentiable on M. Then fx is convex on M if and only the Hessian ∇2fx is positive semi-definite for every x∈M.

Proof: let x\* ∈M be arbitrary . Due to continuity of the second partial derivatives we have that for every λ∈R and y ∈Rn,x\*+λy∈M, there is θ∈0,1 such that

fx\* + λy=fx\*+λ∇fx\*Ty+1/2×λ2yT∇2fx\*+θλyy

⇒ from theorem one we get:

fx\*+λy≥fx\*+λ∇fx\*Ty

so, we can conclude that 

yT∇2fx\*+θλyy≥0

By the limit transition λ ⟶0 we have 

yT∇2fx\*y≥0

⇐ due to positive semi-definitiveness of the Hessian, we have  yT∇2fx\*+θλyy≥0 . Hence:

fx\*+λy≥fx\*+λ∇fx\*Ty

which shows the convexity of f(x) in the view of Theorem one.

# Gradient 
*Figure  SEQ Figure \\* ARABIC 7. The gradient field of function f = x2-4x+y2+2y*
![](Aspose.Words.f2571b5b-5e5e-42a8-b357-02f03b593c59.013.png)![](Aspose.Words.f2571b5b-5e5e-42a8-b357-02f03b593c59.014.jpeg)A gradient simply measures the change in all weights with regard to the change in error. We can also think of a gradient as the slope of a function that has more than one input variable. The higher the gradient, the steeper the slope and the faster a model can learn. But if the slope is zero, the model stops learning. 

## Gradient Descent
Gradient descent is an optimization algorithm used for training a machine learning model. It works on a convex function and changes the function’s parameters iteratively to find the local minimum (Since the function is convex, by finding local minimum we also find the global minimum).

*Figure  SEQ Figure \\* ARABIC 8. The way Gradient Descent works*

![](Aspose.Words.f2571b5b-5e5e-42a8-b357-02f03b593c59.015.png)To find the local minimum, we must take steps proportional to the negative of the gradient (move away from the gradient) of the function at the current point. If we take steps proportional to the positive of the gradient, we will approach a local maximum of the function, and the procedure is called Gradient Ascent.

![](Aspose.Words.f2571b5b-5e5e-42a8-b357-02f03b593c59.016.png)

Gradient descent performs two steps iteratively:

1. **Computes the gradient** (slope), the first order derivative of the function at that point
1. **Moves in the direction opposite to the gradient**

This algorithm can be written as followed:

repeat until convergence{

θj≔θj-α∂∂θjJθ0,θ1

for j=1 and j=0}

Here Alpha is called Learning rate and determines the length of the steps. For gradient descent to reach the local minimum we must set the learning rate to an appropriate value. If alpha is too high, algorithm may not reach the local minimum because it keeps changing. If we set the learning rate to a very small value, gradient descent will eventually reach the local minimum but it may take a while because the function changes very slowly.

*Figure  SEQ Figure \\* ARABIC 9. The effect of learning in Gradient Descent*

![](Aspose.Words.f2571b5b-5e5e-42a8-b357-02f03b593c59.017.png)![](Aspose.Words.f2571b5b-5e5e-42a8-b357-02f03b593c59.018.png)Note: While moving towards the local minima, both gradient and the size of the step decreases. So, the learning rate can be constant over the optimization.

In Figure 10, four different learning rates have been used for a function f. We can make the following observations:

a) Learning rate is optimal, model converges to the minimum

b) Learning rate is too small, it takes more time but converges to the minimum

c) Learning rate is higher than the optimal value, it overshoots but converges ( 1/C < η <2/C)

*Figure  SEQ Figure \\* ARABIC 10. Four different learning rates working on function f*

![](Aspose.Words.f2571b5b-5e5e-42a8-b357-02f03b593c59.019.png)![](Aspose.Words.f2571b5b-5e5e-42a8-b357-02f03b593c59.020.png)d) Learning rate is very large, it overshoots and diverges, moves away from the minima, performance decreases on learning
## Challenges with gradient descent
Although gradient descent is the most common approach for optimization problems, it has its own challenges. Some of them include:
### Local minima and saddle points
In nonconvex problems, gradient descent can have difficulties in finding the global minimum.

*Figure  SEQ Figure \\* ARABIC 11. Local minima and Saddle point can face problems using Gradient Descent*

![](Aspose.Words.f2571b5b-5e5e-42a8-b357-02f03b593c59.021.png)![](Aspose.Words.f2571b5b-5e5e-42a8-b357-02f03b593c59.022.png)In the previous section we mentioned that when the slope of the cost function is at or close to zero, the model stops learning. Other than global minimum, local minima and saddle points can also reach this slope. Local minima have the same shape as the global minimum, where the slope of the cost function increases on either side of the current point. Saddle points, have negative gradient on one side and a non-negative gradient on the other side, causing them to reaching a local maximum on one side and a local minimum on the other. We can use noisy gradients to escape local minimums and saddle points.

### Vanishing and Exploding Gradients
In deeper neural networks we can also encounter two other problems:

Vanishing gradients: This occurs when the gradient is too small. As we move backwards during backpropagation, the gradient continues to become smaller, so the earlier layers in the network learn more slowly than later layers. When this happens, the weight parameters update until they become insignificant and the algorithm stops learning.

Exploding gradients: This happens when the gradient is too large and creates an unstable model. In this case, the model weights will grow too large, and they will eventually be represented as NaN.

# Conclusion
In this lecture we talked about the different types of optimizations in continuous spaces. It has been proven that finding an epsilon-optimal minimizer that has the following relationship with the optimal solution x\*

x\*-x'≤ε

is impossible. However, we introduced a feature called convexity which can help us solve some of these problems. We proved that if a function’s Hessian matrix is positive semi-definitive, it is convex and vice versa. Now if a function is convex, its local minimums are also global. So, by using a decent iterative algorithm, like Gradient Descent, we can find its solutions. But keep in mind that this algorithm can face problems if used on non-convex functions or deeper neural networks

# References
[1] Vandenberghe. <https://web.stanford.edu/~boyd/cvxbook/bv_cvxslides.pdf>

[2] Vandenberghe. <https://web.stanford.edu/class/ee364a/lectures/problems.pdf>

[3] Hladík, Milan. *Discrete and Continuous Optimization*. <https://kam.mff.cuni.cz/~hladik/DSO/text_dso_en.pdf>

[4] Stein, Oliver. *What is continuous optimization?* Institute of Operations Research Karlsruhe Institute of Technology (KIT) <https://kop.ior.kit.edu/downloads/continuous_optimization.pdf>

[5] Patriksson, Michael. Evgrafov, Anton. Andreasson, Niclas. *An Introduction to Continuous Optimization: Foundations and Fundamental Algorithms*. <http://apmath.spbu.ru/cnsa/pdf/monograf/Andreasson,%20Evgrafov,%20Patriksson.%20An%20introduction%20to%20continuous%20optimization.pdf>

[6] Wolkowicz, Henry. *Continuous Optimization and its Applications.*

<https://www.math.uwaterloo.ca/~hwolkowi/henry/reports/talks.d/t06talks.d/06msribirs.d/continuous_optimization.pdf>
