import numpy as np
import pandas as pd

def sigmoid(z):
    return 1/ (1+np.exp(-z)) 


x=np.array((2,1))

W1=np.array([[2,1,-1],
             [-2,-4,1]])

W2= np.array([[-2,3],
              [3,-1],
              [5,0]])

b1=np.array((1,-1,2))
b2= np.array((0,-1))

z1= np.dot(x,W1) + b1
a1= sigmoid(z1)
z2= np.dot(a1,W2) + b2

a2= sigmoid(z2)
print("Activation value is: " , a2)
###################################### Chapter 2 Exercise

# (a)
a0= np.array((1,0,2,-3))
W3= np.array([[4,-5,0,1],
              [-3,6,-1,2],
              [0,1,1,-2],
              [2,0,-3,4]])

W4= np.array([[-2,1],
             [-1,0],
             [1,-3],
             [5,-1]])

b3= np.array((1,-2,0,-1))
b4= np.array((-2,2))

z3= np.dot(a0, W3) +b3
a3=sigmoid(z3)

z4=np.dot(a3, W4) +b4

a4=sigmoid(z4)

print("if (1,0,2,-3) is input into the network then the output obtain is: ", a4)


#(b)\

def relu(z):
    return np.maximum(0,z)


x0= np.array((1,0,2,-3))
W5= np.array([[4,-5,0,1],
              [-3,6,-1,2],
              [0,1,1,-2],
              [2,0,-3,4]])

W6= np.array([[-2,1],
             [-1,0],
             [1,-3],
             [5,-1]])

b5= np.array((1,-2,0,-1))
b6= np.array((-2,2))

z5= np.dot(x0,W5) + b5
a5=relu(z5)
z6=np.dot(a5, W6) + b6
a6= sigmoid(z6)

print("if (1,0,2,-3) is input into the network then the output obtain is: ", a6)


######### Lab 2

def softmax(z):
    return np.exp(z)/np.sum(np.exp(z))

x= np.random.rand(5)

W1= np.random.rand(5, 4)
W2= np.random.rand (4,3)
 
b1= np.random.rand(4)
b2= np.random.rand(3)

z1 = np.dot(x, W1) +b1
a1= relu(z1)

z2= np.dot(a1, W2) +b2
a2= softmax(z2)
print("The activation value is :", a2)



############## 

# number 2

from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder

# Load dataset
iris = load_iris()

print(iris.keys())
#df = pd.DataFrame(iris.data, columns=iris.feature_names)
#df['target'] = iris.target
#print(df.head())

X = iris.data
Y= iris.target
 

print(X[:10])
print(Y[:10])

Y_onehot = np.eye(3)[Y]

print(Y_onehot[:5])