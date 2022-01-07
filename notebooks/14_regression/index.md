# Regression

# Contents

# Introduction
We have some data points and a number or a label is assigned to each data point. Our goal is to predict the number or label of an unseen data point after learning from the data we already have. We assume each data point is a vector $x$ and we want to predict $f(x)$. The first idea is to use interpolation. By using interpolation, we will have a high degree polynomial which fits our training data perfectly. But the problem is that, interpolation leads to overfitting. So the error for unseen data will be too large. In regression, we aim to find the best curve with lower degree. Although there will be some training error here, our test error will decrease since we are avoiding overfitting.

# 1 - What is regression ?

## Problem definition

# 2 - Linear Regression

Here, we want to assign $f(x)$ to each data point $x$. In linear regression, we assume $f$ is a linear function. We can define $f$ as

$$
    f_w(x) = w^T x = \begin{bmatrix}
        w_0 & w_1 & \cdots & w_n
    \end{bmatrix} . \begin{bmatrix}
        1 \\ x_1 \\ \vdots \\ x_n
    \end{bmatrix}.
$$
We assumed $x_0 = 1$ in $x$ to have bias in our function. So the data points are in an n-dimentional space.

We also define $y$ as

$$
    y_w = \begin{bmatrix}
        f_w(x^{(1)}) \\ f_w(x^{(2)}) \\ \vdots \\ f_w(x^{(m)})
    \end{bmatrix}
$$
where $x^{(i)}$ is the i'th data point. 

## Loss Function
After defining $f$, we need to find the best function. By defining a loss function, we can try to minimize loss by changing $w$ in the main function. Assuming $L$ as our loss function, best $f$ will be

$$
    \hat{w} = argmin_w L(y_w, \hat{y}) \rightarrow f_{best}(x) = f_{\hat{w}}(x) 
$$

where $\hat{y}$ is the given number for each data point.

### Mean Squared Error (MSE)
The main loss function we use is mean squared error. It's defined as 

$$
    MSE(y_w, \hat{y}) = \frac{1}{2} \Sigma_{i=1}^m \left[ y_w^{(i)} - \hat{y}^{(i)} \right]^2 .
$$
The main reason for using this function is that we can calculate gradient easily. So we can use gradient descent to find $\hat{w}$.

## Finding $\hat{w}$

### Gradient Descent
We want to use gradient descent to find $\hat{w}$. First, we need to calculate $\nabla_w L(y_w, \hat{y})$ because it's used in the gradient descent method. The partial derivitives for MSE are:

$$
    \frac{\partial MSE}{\partial w_j} = - \Sigma_{i = 1}^m x_j^{(i)} \left[ y_w^{(i)} - \hat{y}^{(i)} \right]
$$

So for gradinet, we have:

$$
    \nabla_w MSE = \begin{bmatrix}
        \frac{\partial MSE}{\partial w_0} \\ \vdots \\ \frac{\partial MSE}{\partial w_n}
    \end{bmatrix}
$$

Now, we can find $\hat{w}$ by the gradient descent algorithm as

$$
    w^{(i+1)} = w^{(i)} - \eta \nabla_{w^{(i)}} L
$$

where $\eta$ is the learning rate.

### Normal Equation
If we define

$$
    X = \begin{bmatrix}
        x^{(1)} \\ \vdots \\ x^{(m)}
    \end{bmatrix}
$$

and solve the equation $\nabla_w MSE = 0$, we get the normal equation. It's defined as
$$
    \hat{w} = (X^TX)^{-1} X^T \hat{y}.
$$
However, for using this equation, our features must be linearly independent. Otherwise, $(X^TX)^{-1}$ is not defined. In that case we can use pseudo inverse of $X^TX$ instead of the $(X^TX)^{-1}$. Since the calculation of inverse is computationaly inefficient, we usually prefer using gradient descent. 

# 3 - Learning Curves Using Polynomials

In this section, we want to find the best $P(x)$ where x is a real number and $P$ is an n'th degree polynomial.

## Reduction to Linear Regression

We can define 
$$
    z = \begin{bmatrix}
        x^0 \\ x^1 \\ \vdots \\ x^n
    \end{bmatrix},
$$
then use linear regression to find $f_w(z) = w^T z$. From definition, we know that $P(x) = f_w(z)$. However, if $n$ is too large, overfitting might happen since we are getting closer and closer to interpolation.

## Overfitting
To prevent overfitting, we must try to define $n$ optimaly. Since $n$ is a hyperparameter here, we can use validation set.

### Using Validation Set (Held-Out Data)
We split a part of the training data, and don't use it for training. Then, by calculating loss function over this data, we can optimize $n$. Since the model hasn't seen these data points, we can be sure that overfitting will decrease.

### Regularization
Although using validation set is a good way to prevent overfitting, We still might be trying to find the curve which best fits the data, not the curve which best matches it. Regularization tries to make $w$ smaller. The intuition is that large coefficients in $w$ happen because it tries to fit the points. It means that the curve will only get closer to each point. However, decreasing coeffiecients will make a better curve which might be further from each point, but does a better job at predicting the unseen data.

We only define $l_2-\text{regularization}$ since it's easier to derive and understand. The loss function is 

$$
    \tilde{E}(y_w, \hat{y}) = \frac{1}{2} \Sigma_{i=1}^m \left[ y_w^{(i)} - \hat{y}^{(i)} \right]^2 + \frac{\lambda}{2}||w||^2
$$
where $\lambda$ is hyperparameter which controls how small the $w$ should be.



********* Examples *************



# 4 - Regularization

## ridge (OPTIONAL)

## lasso (OPTIONAL)

## elastic (OPTIONAL)

# 5 - Logistic Regression



# Conclusion

# Refrences
- https://en.wikipedia.org/wiki/Linear_regression
- https://math.stackexchange.com/questions/1962877/compute-the-gradient-of-mean-square-error
- https://towardsdatascience.com/normal-equation-in-python-the-closed-form-solution-for-linear-regression-13df33f9ad71