import numpy as np
import tensorflow as tf
from math import exp, log, sqrt, pi


class Model:
    
    # initial constructor
    def __init__(self, d, K, r, v, T, n):
        self.d = d # dimension
        self.K= K
        self.r= r
        self.v= v
        self.T =T
        self.n =n
        self.dt =T/n


    # driver
    def f( self, t,  u, Du):
        return -self.r*u


    # numpy eqivalent of f
    def fNumpy( self, t, u, Du):
        return -self.r*u
                            

    # Asian terminal
    def g(self,x):
        total = tf.reduce_mean((tf.reduce_sum(x[:, :, :-1], axis = 2) + tf.reduce_sum(x[:, :, 1:], axis = 2))*0.5*self.dt, axis=1)
        return tf.reduce_max([total - self.K, tf.zeros(tf.shape(x)[0])], axis = 0)


    def gNumpy(self,x):        
        total = np.mean((np.sum(x[:, :, :-1], axis = 2) + np.sum(x[:, :, 1:], axis = 2))*0.5*self.dt, axis=1)
        return np.maximum( total - self.K, 0)
   
