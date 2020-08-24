#PERFORMING IMPORTANCE SAMPLING ON A MONTE CARLO SAMPLE
from scipy.integrate import quad
import math
import numpy as np
n = 10000
sum = 0
import  random
def f_x(x):
    return math.exp(-(x * x)/2)
res, err = quad(f_x,0,5)
print("The numerical result is {:f} (+-{:g})".format(res, err))
def distribution(x):
    dist = np.exp(-x)
    return dist

for i in range(n):
    x = 5 * distribution(i)
    sum += f_x(x)
print("Importance sampled value is:",sum/n)
