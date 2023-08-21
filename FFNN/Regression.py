import sys
import os
import numpy as np
import pandas as pd
np.random.seed(42)

NUM_FEATS = 90
class Net(object):
  def __init__(self, num_layers, num_units):
    self.num_layers = num_layers
    self.num_units = num_units
    self.biases = []
    self.weights = []
    for i in range(num_layers):
      if i==0:
        self.weights.append(np.random.uniform(-1, 1, size=(NUM_FEATS, self.num_units)))
      else:
        self.weights.append(np.random.uniform(-1, 1, size=(self.num_units, self.num_units)))
        self.biases.append(np.random.uniform(-1, 1, size=(self.num_units, 1)))
      self.biases.append(np.random.uniform(-1, 1, size=(1, 1)))
      self.weights.append(np.random.uniform(-1, 1, size=(self.num_units, 1)))
  def __call__(self, X):
    Y_pred = []
    w=self.weights
    b=self.biases
    k=0

    H=[]
    H.append(X)
    
    for i in range(self.num_layers+1):
      A=[]
      if(i==0):
        c=np.maximum(0,np.dot(list(H[0]),w[0]))
        H.append(c)
      elif(i==self.num_layers):
        c=np.dot(list(H[i]),w[i])+b[i-1].T
        H.append(c)
      else:
        c=np.maximum(0,np.dot(list(H[i]),w[i]))+b[i-1].T
        H.append(c)
    Y_pred.append(H[-1])
    return Y_pred[0],H
  def backward(self,y_original,y_pred, HH1, lamda):
    node_d_w,weight_d_w,bias_d_w=[],[],[]
    w=np.einsum('ij,ik->ijk',HH1[-2],(2*(y_pred-y_original)))+2*0.1*self.weights[-1]
    n=np.dot(self.weights[-1],(2*(y_pred-y_original).T)).T
    node_d_w.append(n)
    weight_d_w.append(w)
    bias_d_w.append(2*(y_pred-y_original))
    for i in range(self.num_layers-1):
      bias_d_w.append(n)
      w=np.einsum('ij,ik->ijk',node_d_w[i],HH1[self.num_layers-i-1])+2*0.01*self.weights[self.num_layers-i-1]
      n=node_d_w[i]@self.weights[self.num_layers-i-1]
      node_d_w.append(n)
      weight_d_w.append(w)
    w=np.einsum('ij,ik->ijk',HH1[0],node_d_w[-1])+2*0.01*self.weights[0]
    weight_d_w.append(w)
    return weight_d_w,node_d_w,bias_d_w
  
class Optimizer(object):
  def __init__(self, learning_rate):
    print("hello")
  def step(self, weights, biases, delta_weights, delta_biases):
    print("hello")
def loss_mse(y, y_hat):
  print(np.sum(np.square(y_hat-y))/len(y))
def loss_regularization(weights, biases):
  return np.square(weights[-1])
def cross_entropy_loss(y, y_hat):
  print(np.mean(-np.log(y_hat[range(len(y_hat)), np.argmax(y,axis=1)])))