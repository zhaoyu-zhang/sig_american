# create all the networks necessary
import numpy as np
import tensorflow as tf
from tensorflow.contrib.slim import fully_connected as fc

# Feed forward classical
class FeedForwardUDU:

    def __init__(self, d,  sig_d, layerSize, activation):
        self.d=d
        self.sig_d = sig_d
        self.layerSize = layerSize
        self.activation= activation


    def createNetwork(self, X,iStep):
        with tf.variable_scope("NetWork"+str(iStep) , reuse=tf.AUTO_REUSE):
            fPrev= fc(X, self.layerSize[0], scope='enc_fc1', activation_fn=self.activation)
            for i in np.arange(len(self.layerSize)-1):
                scopeName='enc_fc'+str(i+2)
                f = fc(fPrev,self.layerSize[i+1], scope=scopeName, activation_fn=self.activation)
                fPrev = f
            UZ  = fc(fPrev,self.d+1, scope='uPDu',activation_fn= None)
        return  UZ[:,0], UZ[:,1:]


    def createNetworkWithInitializer(self, X, iStep, weightInit, biasInit):
        with tf.variable_scope("NetWork"+str(iStep) , reuse=tf.AUTO_REUSE):
            cMinW =0
            cMinB= 0
            fPrev= fc(X, self.layerSize[0], scope='enc_fc1', activation_fn=self.activation,  weights_initializer=tf.constant_initializer(np.reshape(weightInit[:self.sig_d*self.layerSize[0]],[self.sig_d,self.layerSize[0]])), biases_initializer= tf.constant_initializer(biasInit[0:self.layerSize[0]]) )
            cMinW += self.d*self.layerSize[0]
            cMinB +=  self.layerSize[0]
            for i in np.arange(len(self.layerSize)-1):
                scopeName='enc_fc'+str(i+2)
                f = fc(fPrev,self.layerSize[i+1], scope=scopeName, activation_fn=self.activation,  weights_initializer=tf.constant_initializer(np.reshape(weightInit[cMinW : cMinW+self.layerSize[i]*self.layerSize[i+1]],[self.layerSize[i],self.layerSize[i+1]])) , biases_initializer= tf.constant_initializer(biasInit[cMinB:cMinB+self.layerSize[i+1]]))
                cMinW+= self.layerSize[i]*self.layerSize[i+1]
                cMinB+= self.layerSize[i+1]
                fPrev = f
            UDU  = fc(fPrev,self.d+1, scope='uPDu',activation_fn= None, weights_initializer=tf.constant_initializer(np.reshape(weightInit[cMinW:cMinW+self.layerSize[len(self.layerSize)-1]*(self.d+1)],[self.layerSize[len(self.layerSize)-1],self.d+1])), biases_initializer= tf.constant_initializer(biasInit[cMinB:cMinB+(self.d+1)]))
        return  UDU[:,0], UDU[:,1:]


    # get back weights and bias associated to the network
    def getBackWeightAndBias(self,iStep):
        Weights = []
        Bias = []
        with tf.variable_scope("NetWork"+str(iStep), reuse=tf.AUTO_REUSE):
            Weights.append(tf.get_variable("enc_fc1/weights", [self.sig_d,self.layerSize[0]]))
            Bias.append(tf.get_variable("enc_fc1/biases", [self.layerSize[0]]))
            for i in np.arange(len(self.layerSize)-1):
                scopeName='enc_fc'+str(i+2)
                Weights.append(tf.get_variable(scopeName+"/weights", [self.layerSize[i],self.layerSize[i+1]]))
                Bias.append(tf.get_variable(scopeName+"/biases", [self.layerSize[i+1]]))
            Weights.append(tf.get_variable("uPDu/weights",[self.layerSize[len(self.layerSize)-1],self.d+1]))
            Bias.append(tf.get_variable("uPDu/biases",[self.d+1]))
        return Weights, Bias


    # transfrom the list of weight in a single weight array (idem for bias)
    def getWeights( self, sess, weightLoc, biasLoc):
        # get back weight
        weights =sess.run(weightLoc)
        bias =  sess.run(biasLoc)
        return np.concatenate([ x.flatten() for x in weights]), np.concatenate([x.flatten() for x in bias])
        


