# This file includes operations on numpy

# Python default operation is very slow.
import numpy as np
np.random.seed(0)

def compute_reciprocals(values):
    output = np.empty(len(values))
    for i in range(len(values)):
        output[i] = 1.0 / values[i]
    return output

values = np.random.randint(1,10,size=5)
compute_reciprocals(values)

big_array = np.random.randint(1, 100, size=1000000)
%timeit compute_reciprocals(big_array)
# time it
%timeit 1/big_array

# numpy operations take ms to complete


# Array arithmetic
x = np.arange(4)
print("x   ", x)
print("x + 5 ", x+5)
abs(x) # to get absolute value of x

x = [1,2,3]
np.exp(x)     e^x
np.exp2(x)    2^x
np.power(3,x) 3^x


x = [1,2,3,4,10]
np.log(x)
np.log2(x)
np.log10(x)



# Aggregating functions
L = np.random.random(100)
#python version to sum
sum(L)

#numpy version to sum
np.sum(L)


min(big_array), max(big_array)
np.min(big_array), np.max(big_array)

M = np.random.random((3,4))
print(M)

# sum all the numbers in matrix
M.sum()

# adding column wise
M.min(axis=0)

# adding row wise
M.max(axis=1)

# np.mean, np.std, np.var, np.argmin, np.argmax

#####################################################################
# Broadcasting Numpy array
import numpy as np
a = np.array([0,1,2])
b = np.array([5,5,5])

a + b

a + 5 #  broadcasting

# Rule of broadcasting
(3,2)
(3)

It will be first converted to
(3,2)
(1,3)

then it starts from left and multiply the dimension
(3,2)
(3,3)

Since 2,3 doesn't match it's an error




