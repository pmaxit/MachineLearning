#python list is more than the list
L = list(range(10))

# python create dynamic typing. We can create hetrogenous lists
L3 = [True, "2", 3.0, 4]
print [type(item) for item in L3]

# flexibility comes with the cost. Each object in the list is python object which contains the type of the object and the value.

# Numpy array is designed to store same type data with efficient operations
import numpy as np

# Creating Arrays from python list
np.array([1,4,2,5,3])

np.array([3.14, 4, 2, 3])
# every number will be converted to float

np.array([1,2,3,4], dtype ='float32')

np.array([range(i, i+3) for i in [2,4,6]])


# Creating Arrays from scratch
np.zeros(10, dtype='int')

# Create a 3X5 floating point array filled with ones
np.ones((3,5), dtype=float)

# create a 3X5 array filled with 3.14
np.full((3,5), 3.14)

# create an array filled with linear sequence starting at 0 , ending at 20, stepping by 2
np.arange(0, 20, 2)

# Create an array of five values evenly spaced between 0 and 1
np.linspace(0, 1, 5)

# Create a 3X3 array of uniformly distributed random values between 0 and 1
np.random.random((3,3))
# normally distributed values
np.random.normal((3,3))

np.eye(3)


# Numpy standard data types
#Numpy array contain values of single type

np.zeros(10, dtype='int16')

#########################################################################
import numpy as np
np.random.seed(0)  # seed for reproducibility

x1 = np.random.randint(10, size=6)  # One-dimensional array
x2 = np.random.randint(10, size=(3, 4))  # Two-dimensional array
x3 = np.random.randint(10, size=(3, 4, 5))  # Three-dimensional array

print("x3 ndim: ", x3.ndim)
print("x3 shape:", x3.shape)
print("x3 size: ", x3.size)
print("X3 type: ", x3.dtype)

ndim : 3
shape : 3,4,5
size : 60

#x3.dtype => int64

# Accessing elements
x1[0]
X1[4]
X1[-1]

# For multi dimensional array
x2[0,0]
x2[2,0]


# Array Slicing
x = np.arange(10)
x[:5]
array([0,1,2,3,4])
x[::2]
array([0,2,4,6,8])
x[::-1]
array([9,8,7,6,5,4,3,2,1,0])

x[5::-2]
array([5,3,1])

# Multi dimensional sub array
x2
x2[:2,:3] # two rows and three columns
x2[:3, ::2] # all rows and every other column
x2[::-1,::-1]

# Accessing first column of x2
print (x2[:,0])

# First row of x2
print (x2[0,:])

x2_sub = x2[:2,:2)
print(x2_sub)
[[12 5]
 [ 7 6 ]]

x2_sub[0,0] = 99
# original array is changed too.

# But if we need to create copy of arrays
x2_sub_copy = x2[:2,:2].copy()
x2_sub_copy[0,0] = 99

# original array will not change

# Reshaping of Arrays
grid = np.arange(1,10).reshape(3,3)
1 2 3
4 5 6
7 8 9

x = np.array([1,2,3])
x.reshape((1,3))

# row vector via reshape
x.reshape((1,3))
# or we can use newaxis
x[np.newaxis,:]

# Array concatenation and splitting
# Concatenation of arrays
x = np.array([1,2,3])
y = np.array([3,2,1])
np.concatenate([x,y])
[1,2,3,3,2,1]

np.concatenate([grid,grid], axis=1]) # adding columns

# for working with arrays of mixed dimension, it can be clearer to use the np.vstack and np.hstack functions
x = np.array([1,2,3])
grid = np.array([[9, 8, 7], [6,5,4]])
np.vstack([x,grid])

# Splitting of arrays
x = [ 1,2,3,4,5 ,6 ,7]
x1, x2, x3 = np.split(x, [3,5])

upper, lower = np.vsplit(grid, [2])
left, right = np.hsplit(grid,[2])



