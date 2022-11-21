# Prce Bermudan Option with Signature
# A session is open and a graph is created and stored at each time step.
# based on BSDE discretization with an Euler scheme

import numpy as np
import tensorflow as tf
import sys, traceback
import math
from tensorflow.contrib.slim import fully_connected as fc
import time
from tensorflow.python.tools import inspect_checkpoint as chkp
import matplotlib.pyplot as plt
import pylab as P
from mpl_toolkits.mplot3d import Axes3D


class Base:

    # initial constructor
    def __init__(self, xInit, model, K, r, v, T, nbStep, nbSeg, batch_size, network, initialLearningRate=5e-4, graphLoc=""):
        self.model = model
        self.d = len(xInit)  # dimension
        self.xInit = xInit
        self.K = K
        self.r = r
        self.v = v
        self.T = T
        self.segs = nbSeg
        self.nbStep = nbStep
        self.TStep = T / nbStep
        self.TSeg = T / nbSeg
        self.initialLearningRate = initialLearningRate
        self.network = network
        self.graphLoc = graphLoc
        self.batchSize = batch_size
        self.Paths_feed = np.load(
            "Paths_{}_dim_{}_r_{}_sigma_{}_x_{}_K_{}.npy".format(self.nbStep, self.d, self.r, self.v, self.xInit[0],
                                                                 self.K))[batch_size:]
        self.Paths_test = np.load(
            "Paths_{}_dim_{}_r_{}_sigma_{}_x_{}_K_{}.npy".format(self.nbStep, self.d, self.r, self.v, self.xInit[0],
                                                                 self.K))[:batch_size]
        self.W_feed = np.load("W_{}_from_{}_dim_{}.npy".format(self.segs, self.nbStep, self.d))[batch_size:]
        self.W_test = np.load("W_{}_from_{}_dim_{}.npy".format(self.segs, self.nbStep, self.d))[:batch_size]
        self.signature_test = np.load(
            "sig_{}_from_{}_dim_{}_r_{}_sigma_{}_x_{}_K_{}.npy".format(self.segs, self.nbStep, self.d, self.r, self.v,
                                                                       self.xInit[0], self.K))[:batch_size]
        self.signature_feed = np.load(
            "sig_{}_from_{}_dim_{}_r_{}_sigma_{}_x_{}_K_{}.npy".format(self.segs, self.nbStep, self.d, self.r, self.v,
                                                                       self.xInit[0], self.K))[batch_size:]
        self.sig_dim = self.signature_feed.shape[-1]


    # Step 0 build
    def build0(self):
        self.learning_rate = tf.placeholder(tf.float32, shape=[], name="learning_rate")
        self.BrownSigT = tf.placeholder(dtype=tf.float32, shape=[None, self.d], name='DWT')
        self.UNext = tf.placeholder(dtype=tf.float32, shape=[None], name='U')
        sample_size = tf.shape(self.BrownSigT)[0]
        self.Y0 = tf.get_variable("Y0", [], tf.float32, self.Y0_initializer)
        self.Z0 = tf.get_variable("Z0", None, tf.float32,
                                  tf.random_uniform([1, self.d], minval=-.05, maxval=.05, dtype=tf.float32, seed=2))
        Y = self.Y0 * tf.ones([sample_size])
        Z = tf.tile(self.Z0, [sample_size, 1])
        Y = Y - self.model.f(0., Y, Z) * self.TSeg + tf.reduce_sum(tf.multiply(Z, self.BrownSigT), axis=1)
        self.total_loss = tf.reduce_sum(tf.pow(self.UNext - Y, 2)) / self.batchSize
        # optimizer
        self.train = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.total_loss)


    # general build
    def build(self, iSeg):
        self.learning_rate = tf.placeholder(tf.float32, shape=[], name="learning_rate")
        self.XPrev = tf.placeholder(dtype=tf.float32,
                                    shape=[None, self.d, int((iSeg - 1) / self.segs * self.nbStep + 1)], name='XPrev')
        self.signaturePrev = tf.placeholder(dtype=tf.float32, shape=[None, self.sig_dim])
        self.BrownSigT = tf.placeholder(dtype=tf.float32, shape=[None, self.d], name='DWT')  # sigma dt DW
        self.UNext = tf.placeholder(dtype=tf.float32, shape=[None], name='U')
        ULoc, DULoc = self.network.createNetworkWithInitializer(self.signaturePrev, iSeg, self.CWeight, self.Cbias)
        # stores
        self.U = ULoc
        self.Z = DULoc
        Y = ULoc - self.model.f(self.TSeg * iSeg, ULoc, DULoc) * self.TSeg + tf.reduce_sum(
            tf.multiply(DULoc, self.BrownSigT), axis=1)
        # Loss
        self.total_loss = tf.reduce_sum(tf.pow(self.UNext - Y, 2)) / self.batchSize
        # optimizer
        self.train = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.total_loss)


    # last step
    def buildLast(self, iSeg):
        self.learning_rate = tf.placeholder(tf.float32, shape=[], name="learning_rate")
        self.X = tf.placeholder(dtype=tf.float32, shape=[None, self.d, self.nbStep + 1], name='X')
        self.signaturePrev = tf.placeholder(dtype=tf.float32, shape=[None, self.sig_dim], name='SigPrev')
        self.BrownSigT = tf.placeholder(dtype=tf.float32, shape=[None, self.d], name='DWT')  # sigma dt DW
        ULoc, DULoc = self.network.createNetwork(self.signaturePrev, iSeg)
        # store U
        self.U = ULoc
        self.Z = DULoc
        # Y
        Y = ULoc - self.model.f(self.TSeg * (self.segs - 1), ULoc, DULoc) * self.TSeg + tf.reduce_sum(tf.multiply(DULoc, self.BrownSigT), axis=1)
        g_final = tf.reshape(self.model.g(self.X), tf.shape(Y))
        # Loss
        totalLossLoc = tf.reduce_sum(tf.pow(Y - g_final, 2)) / self.batchSize
        self.total_loss = totalLossLoc
        # optimizer
        self.train = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(totalLossLoc)



    def BuildAndtrainML(self, num_epoch=100, num_epochExt=100, num_epochExt1=4, nbOuterLearning=20,
                        min_decrease_rate=0.05):

        tf.reset_default_graph()

        # begin with last step
        ######################
        gPrev = tf.Graph()
        # print("Begin  session last step")
        with gPrev.as_default():
            # build
            self.sess = tf.Session()
            # initialize
            self.theLearningRate = self.initialLearningRate
            self.buildLast(self.segs - 1)
            # graph for weights
            self.weightLoc, self.biasLoc = self.network.getBackWeightAndBias(self.segs - 1)
            # initialize variables
            self.sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            # history loss
            loss_hist_stop_criterion = []

            for iout in range(num_epochExt):
                for epoch in range(num_epoch):
                    index = np.random.randint(self.W_feed.shape[0], size=self.batchSize)
                    wSigT = self.W_feed[index, :, self.segs].reshape([self.batchSize, self.d]) - self.W_feed[index, :,
                                                                                                 self.segs - 1].reshape(
                        [self.batchSize, self.d])
                    x = self.Paths_feed[index, :, :self.nbStep + 1].reshape([self.batchSize, self.d, self.nbStep + 1])
                    signaturePrev = self.signature_feed[index, self.segs, :].reshape([self.batchSize, self.sig_dim])
                    feed_dict = {self.X: x, self.BrownSigT: wSigT, self.learning_rate: self.theLearningRate,
                                 self.signaturePrev: signaturePrev}
                    self.sess.run(self.train, feed_dict)
                wSigTVal = self.W_test[:, :, self.segs].reshape([self.batchSize, self.d]) - self.W_test[:, :,
                                                                                            self.segs - 1].reshape(
                    [self.batchSize, self.d])
                xVal = self.Paths_test[:, :, :self.nbStep + 1].reshape([self.batchSize, self.d, self.nbStep + 1])
                signatureValPrev = self.signature_test[:, self.segs - 1, :].reshape([self.batchSize, self.sig_dim])

                test_dict = {self.X: xVal, self.BrownSigT: wSigTVal, self.learning_rate: self.theLearningRate,
                             self.signaturePrev: signatureValPrev}
                valLoss = self.sess.run([self.total_loss], test_dict)
                loss_hist_stop_criterion.append(valLoss)
                if (iout % nbOuterLearning == 0):
                    mean_loss_from_last_check = np.mean(loss_hist_stop_criterion)
                    if (iout > 0):
                        decrease_rate = (last_loss_check - mean_loss_from_last_check) / last_loss_check
                        if decrease_rate < min_decrease_rate:
                            self.theLearningRate = np.maximum(1e-3, self.theLearningRate * .85)
                    last_loss_check = mean_loss_from_last_check
                    loss_hist_stop_criterion = []

        # get back weight
        self.CWeight, self.Cbias = self.network.getWeights(self.sess, self.weightLoc, self.biasLoc)
        # save
        if (self.graphLoc != ""):
            save_path = saver.save(self.sess, self.graphLoc + str(self.segs - 1))
        # save old session
        sessPrev = self.sess
        # sAVE u and placeholder used
        UPrev = self.U
        signaturePrev_prev = self.signaturePrev

        # backward resolutions
        #####################
        for iseg in range(self.segs - 2, 0, -1):  # 8 7  --- 1
            # new graph
            gCurr = tf.Graph()
            with gCurr.as_default():
                self.sess = tf.Session()
                # initialize
                self.theLearningRate = self.initialLearningRate
                # build
                start_time = time.time()
                self.build(iseg)
                # graph for weights
                self.weightLoc, self.biasLoc = self.network.getBackWeightAndBias(iseg)
                # initialize variables
                self.sess.run(tf.global_variables_initializer())
                # save
                saver = tf.train.Saver()

                # history loss
                loss_hist_stop_criterion = []
                # back in time
                for iout in range(num_epochExt1):
                    loss_all = []
                    for epoch in range(num_epoch):
                        index = np.random.randint(self.W_feed.shape[0], size=self.batchSize)
                        wSigT = self.W_feed[index, :, iseg + 1].reshape([self.batchSize, self.d]) - self.W_feed[index,
                                                                                                    :, iseg].reshape(
                            [self.batchSize, self.d])
                        x = self.Paths_feed[index, :, :int((iseg + 1) / self.segs * self.nbStep + 1)].reshape(
                            [self.batchSize, self.d, int((iseg + 1) / self.segs * self.nbStep + 1)])
                        signaturePrev = self.signature_feed[index, iseg, :].reshape([self.batchSize, self.sig_dim])
                        signature = self.signature_feed[index, iseg + 1, :].reshape([self.batchSize, self.sig_dim])

                        UNext = sessPrev.run(UPrev, feed_dict={signaturePrev_prev: signature})
                        Exercise = np.reshape(self.model.gNumpy(x), np.shape(UNext))
                        UNext = np.maximum(UNext, Exercise)
                        feed_dict = {self.UNext: UNext, self.BrownSigT: wSigT, self.learning_rate: self.theLearningRate,
                                     self.signaturePrev: signaturePrev}
                        self.sess.run(self.train, feed_dict)
                        loss_temp = self.sess.run(self.total_loss, feed_dict)
                        loss_all.append(loss_temp)

                    wSigTVal = self.W_test[:, :, iseg + 1].reshape([self.batchSize, self.d]) - self.W_test[:, :,
                                                                                               iseg].reshape(
                        [self.batchSize, self.d])
                    xVal = self.Paths_test[:, :, :int((iseg + 1) / self.segs * self.nbStep + 1)].reshape(
                        [self.batchSize, self.d, int((iseg + 1) / self.segs * self.nbStep + 1)])
                    signatureValPrev = self.signature_test[:, iseg, :].reshape([self.batchSize, self.sig_dim])
                    signatureVal = self.signature_test[:, iseg + 1, :].reshape([self.batchSize, self.sig_dim])

                    # evaluate at next step
                    UNext = sessPrev.run(UPrev, feed_dict={signaturePrev_prev: signatureVal})
                    Exercise = np.reshape(self.model.gNumpy(xVal), np.shape(UNext))
                    UNext = np.maximum(UNext, Exercise)

                    test_dict = {self.UNext: UNext, self.BrownSigT: wSigTVal, self.learning_rate: self.theLearningRate,
                                 self.signaturePrev: signatureValPrev}
                    valLoss = self.sess.run([self.total_loss], test_dict)
                    loss_hist_stop_criterion.append(valLoss)
                    if (iout % nbOuterLearning == 0):
                        mean_loss_from_last_check = np.mean(loss_hist_stop_criterion)
                        if (iout > 0):
                            decrease_rate = (last_loss_check - mean_loss_from_last_check) / last_loss_check
                            if decrease_rate < min_decrease_rate:
                                self.theLearningRate = np.maximum(1e-3, self.theLearningRate * .85)
                        last_loss_check = mean_loss_from_last_check
                        loss_hist_stop_criterion = []
                # save
                if (self.graphLoc != ""):
                    save_path = saver.save(self.sess, self.graphLoc + str(iseg))
            # get back weight
            self.CWeight, self.Cbias = self.network.getWeights(self.sess, self.weightLoc, self.biasLoc)
            # switch graph, close old session
            gPrev = gCurr
            sessPrev.close()
            # new old session and U stored
            sessPrev = self.sess
            # sAVE u
            UPrev = self.U
            signaturePrev_prev = self.signaturePrev

        # last run
        # new graph
        gCurr = tf.Graph()
        with gCurr.as_default():
            self.sess = tf.Session()
            # initialize
            self.theLearningRate = self.initialLearningRate
            # build
            # initial value for Y0 expectation of UN
            index = np.random.randint(self.W_feed.shape[0], size=self.batchSize)
            signature = self.signature_feed[index, 1, :].reshape([self.batchSize, self.sig_dim])
            UNext = sessPrev.run(UPrev, feed_dict={signaturePrev_prev: signature})
            Y0_init = np.mean(UNext)
            self.Y0_initializer = tf.constant_initializer(Y0_init)
            # build graph
            self.build0()
            # initialize variables
            self.sess.run(tf.global_variables_initializer())
            # saver
            saver = tf.train.Saver()

            # history loss
            loss_hist_stop_criterion = []  # history loss

            for iout in range(num_epochExt1):
                loss_all = []
                for epoch in range(num_epoch):
                    index = np.random.randint(self.W_feed.shape[0], size=self.batchSize)
                    wSigT = self.W_feed[index, :, 1].reshape([self.batchSize, self.d]) - self.W_feed[index, :,
                                                                                         0].reshape(
                        [self.batchSize, self.d])
                    x = self.Paths_feed[index, :, :int((1) / self.segs * self.nbStep + 1)].reshape(
                        [self.batchSize, self.d, int((1) / self.segs * self.nbStep + 1)])
                    signature = self.signature_feed[index, 1, :].reshape([self.batchSize, self.sig_dim])
                    UNext = sessPrev.run(UPrev, feed_dict={signaturePrev_prev: signature})
                    Exercise = np.reshape(self.model.gNumpy(x), np.shape(UNext))
                    UNext = np.maximum(UNext, Exercise)
                    feed_dict = {self.UNext: UNext, self.BrownSigT: wSigT, self.learning_rate: self.theLearningRate}

                    # estimate U at x (next step)
                    self.sess.run(self.train, feed_dict)
                    loss_all = []
                    loss_all.append(loss_temp)

                wSigTVal = self.W_test[:, :, 1].reshape([self.batchSize, self.d]) - self.W_test[:, :, 0].reshape(
                    [self.batchSize, self.d])
                xVal = self.Paths_test[:, :, :int((1) / self.segs * self.nbStep + 1)].reshape(
                    [self.batchSize, self.d, int((1) / self.segs * self.nbStep + 1)])
                signatureVal = self.signature_test[:, 1, :].reshape([self.batchSize, self.sig_dim])
                UNext = sessPrev.run(UPrev, feed_dict={signaturePrev_prev: signatureVal})
                Exercise = np.reshape(self.model.gNumpy(xVal), np.shape(UNext))
                UNext = np.maximum(UNext, Exercise)
                test_dict = {self.UNext: UNext, self.BrownSigT: wSigTVal}
                valLoss = self.sess.run([self.total_loss], test_dict)

                loss_hist_stop_criterion.append(valLoss)
                if (iout % nbOuterLearning == 0):
                    mean_loss_from_last_check = np.mean(loss_hist_stop_criterion)
                    if (iout > 0):
                        decrease_rate = (last_loss_check - mean_loss_from_last_check) / last_loss_check
                        if decrease_rate < min_decrease_rate:
                            self.theLearningRate = np.maximum(1e-3, self.theLearningRate * .85)
                    last_loss_check = mean_loss_from_last_check
                    loss_hist_stop_criterion = []

            Y0, Z0 = self.sess.run([self.Y0, self.Z0])
            if (self.graphLoc != ""):
                save_path = saver.save(self.sess, self.graphLoc + "0")

        return Y0, Z0
