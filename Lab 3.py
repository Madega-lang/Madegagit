import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
def sigmoid(z):
    return 1/(1+np.exp(-z))

def Relu(z):
    return np.max(0, z)

def softmax(z):
    return np.exp(z)/(np.sum(np.exp(z)))

######### Exercise
# 2 (a)

x=np.array((1,-2))
t=np.array((1,0))

W1= np.array([[1,-1],
              [-3,-2]])

W2= np.array([[-2,0],
              [0,3]])

b1= np.array((0,1))
b2= np.array((1,-2))

z1= np.dot(x, W1) +b1
a1= sigmoid(z1)
z2= np.dot(a1, W2) + b2

y=a2= sigmoid(z2)

print("For input (1,-2) the output are given as: ", a2)

# (b)

loss_before= 0.5*np.sum((y-t)**2)

print(f"{loss_before:.4}")

# (c)
lr= 0.1
delta2= (a2-t)*a2*(1-a2)
delta1 = np.dot(W2, delta2)*a1*(1-a1)

W2_new = W2- lr* np.outer(a2, delta2)
W1_new = W1- lr*np.outer(x, delta1)

b2_new = b2 - lr*delta2
b1_new = b1 -lr*delta1

print("The updated weights on the output layer is:", W2_new, "and", b2_new)
print("The updated weights on the input layer is:", W1_new, "and", b1_new)

# (d)

z3 =np.dot(x,W1_new) +b1_new
a3 =sigmoid(z3)
z4 =np.dot(a3, W2_new)+ b2_new
y1=a4= sigmoid(z4)

loss_new = 0.5 * np.sum((y1-t)**2)


print("For input (1,-2) the output under backpropagation are given as: ", a4)
print("The loss after backpropagation is given by:", f"{loss_new:.4}")


# 2
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder

iris= load_iris()

print(iris.keys())

X=iris.data
Y= iris.target
# Hot ecoding
Y_hot= np.eye(3)[Y]
print(Y_hot[:5])

# Initialze Weight
np.random.seed(123)

W_1 = np.random.rand(4,10)* 0.1
W_2 = np.random.rand(10,3)*0.1
b_1 = np.random.rand(10)
b_2= np.random.rand(3)

# Training 

epoch =50
lr=0.01



for epoch in range(epoch):

    total_loss= 0
    indices =np.random.permutation(len(X))
    X_Suffled =X[indices]
    Y_suffled= Y_hot[indices]


    
    for i in range(len(X_Suffled)): 
      
      x=X_Suffled[i]
      t=Y_suffled[i]

     #Feedforward
      z1=np.dot(x, W_1) +b_1
      a1=sigmoid(z1)
      z2=np.dot(a1, W_2) +b_2
      y=a2= softmax(z2)
      #Sum of Square loss
      Loss= 0.5 * np.sum((y-t)**2)
      total_loss += Loss

      # Backpropagation
      Delta_2= (a2-t)
      Delta_1 = np.dot(W_2, Delta_2) *a1*(1-a1)
      W2_updated = W_2 - lr* np.outer(a1, Delta_2)
      W1_updated = W_1 -lr* np.outer(x, Delta_1)
      b1_Up= b_1 - lr*Delta_1
      b2_Up= b_2 -lr*Delta_2

print(f"Epoch {epoch+1}, Loss={total_loss:.4f}")     


    

 
 
    