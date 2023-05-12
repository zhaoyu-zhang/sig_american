import numpy as np
import tensorflow as tf
from math import exp, log, sqrt, pi


class Model:
    
    # initial constructor
    def __init__(self, d, K, r, v, T, n, n_tilde, delay):
        self.d = d # dimension
        self.K= K
        self.r= r
        self.v= v
        self.T =T
        self.n =n
        self.n_tilde = n_tilde
        self.dt =T/n
        self.delay = delay


    # driver
    def f( self, t,  u, Du):
        return -self.r*u


    # numpy eqivalent of f
    def fNumpy( self, t, u, Du):
        return -self.r*u

    # driver delay
    def f_delay( self, t,  u, Du):
        return tf.zeros(1)


    # numpy eqivalent of f
    def fNumpy_delay( self, t, u, Du):
        return 0
                            

    # Asian terminal
    def g(self,x):
        # compute the integral
        total = tf.reduce_mean((tf.reduce_sum(x[:, :, :-1], axis = 2) + tf.reduce_sum(x[:, :, 1:], axis = 2))*0.5*self.dt, axis=1)
        t = int((x.shape[2]-1))/self.n*self.T
        # this is the average over time
        total = total/t
        return tf.reduce_max([total - self.K, tf.zeros(tf.shape(x)[0])], axis = 0)

    def gNumpy(self,x):        
        total = np.mean((np.sum(x[:, :, :-1], axis = 2) + np.sum(x[:, :, 1:], axis = 2))*0.5*self.dt, axis=1)
        t = int((x.shape[2]-1))/self.n*self.T
        total = total / t
        return np.maximum( total - self.K, 0)


    def g_Bermudan(self,x):
        cut_points = np.linspace(0, self.n, self.n_tilde + 1, dtype=int)
        dt = 1 / self.n
        dseg = 1 / self.n_tilde
        prod = 1
        for i in range(len(cut_points) - 1):
            cut = x[:, :, cut_points[i]:cut_points[i + 1]+1]
            # product of the integrals
            prod = prod * (tf.reduce_sum(cut[:, :, :-1], axis=2) + tf.reduce_sum(cut[:, :, 1:], axis=2)) * 0.5 * dt / dseg
        # geometric average
        total = tf.reduce_mean(prod ** (1 / self.n_tilde), axis=1)
        return tf.reduce_max([self.K - total, tf.zeros(tf.shape(x)[0])], axis = 0)

    def gNumpy_Bermudan(self,x, iseg):
        cut_points = np.linspace(0, int((iseg+1)*self.n/self.n_tilde), (iseg+ 1)+1, dtype=int) # iseg from 8
        dt = 1 / self.n
        dseg = 1 / self.n_tilde
        prod = 1
        for i in range(len(cut_points) - 1):
            cut = x[:, :, cut_points[i]:cut_points[i + 1]+1]
            prod = prod * (np.sum(cut[:, :, :-1], axis=2) + np.sum(cut[:, :, 1:], axis=2)) * 0.5 * dt / dseg
        total = np.mean(prod ** (1 / (iseg+1)), axis=1)
        return np.maximum( self.K - total, 0)


    def g_delay(self,x):
        delay_step = int(self.delay / self.T * self.n)
        if int((x.shape[2] - 1)) < delay_step:
            return np.zeros((x.shape[0], 1))
        else:
            return x[:, :, int((x.shape[2] - 1)) - delay_step]

    def gNumpy_delay(self,x):
        delay_step = int(self.delay / self.T * self.n)
        if int((x.shape[2] - 1)) < delay_step:
            return np.zeros((x.shape[0], 1))
        else:
            return x[:, :, int((x.shape[2] - 1)) - delay_step]
   
