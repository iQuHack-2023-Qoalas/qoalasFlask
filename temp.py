## Cartesian example: electrons and lattice

import covalent as ct
import math

@ct.electron
def add(x, y):
    return x + y

@ct.electron
def square(x):
    return x**2

@ct.electron
def sqroot(x):
    return math.sqrt(x)

@ct.lattice ## Compute the Cartesian distance between two points in 2D
def cart_dist(x=0, y=0):
    x2 = square(x)
    y2 = square(y)
    sum_xy = add(x2, y2)
    return sqroot(sum_xy)

id = ct.dispatch(cart_dist)()
result = ct.get_result(id)
print(id)
print(result)