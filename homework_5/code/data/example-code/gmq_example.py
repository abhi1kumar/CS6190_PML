'''
This file contains sample code about how to use Gauss–Hermite quadrature to compute a specific type of integral numerically.

The general form of this type of integral is:( see https://en.wikipedia.org/wiki/Gauss%E2%80%93Hermite_quadrature for more details)

F = int_{ -inf}^{+inf} e^{-x*x) f(x) dx,  (1)

in which we're calculating the integral of f(x) in the range ( -inf, +inf) weighted by e^(-x*x ).
Note that for f(x) being polynomial function, this integral is guaranteed to converge. But for some others convergence is not guaranteed.
'''

import numpy as np


def gass_hermite_quad( f, degree):
    '''
    Calculate the integral (1) numerically.
    :param f: target function, takes a array as input x = [x0, x1,...,xn], and return a array of function values f(x) = [f(x0),f(x1), ..., f(xn)]
    :param degree: integer, >=1, number of points
    :return:
    '''

    points, weights = np.polynomial.hermite.hermgauss( degree)

    #function values at given points
    f_x = f( points)

    #weighted sum of function values
    F = np.sum( f_x  * weights)

    return F

if __name__ == '__main__':

    # Example 1, f(x) = x^2, degree = 3, whose closed form solution is sqrt(pi) / 2
    def x_square( x):
        return x* x
    F = gass_hermite_quad( x_square,3 )
    print( F)

    # Example 2, f(x) = x * sin x, degree = 10, whose closed form solution is sqrt( pi) / e^(1/4) / 2)
    def my_func(x):
        return x * np.sin( x)

    F = gass_hermite_quad( my_func, degree=10)
    print(F)









