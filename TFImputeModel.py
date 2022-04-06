""" 
@author Jianxing Feng
"""
import numpy as np
import theano
import theano.tensor as T
from sklearn.base import BaseEstimator
import logging
import time
import os
import datetime
#import cPickle as pickle
import pickle
from collections import OrderedDict
from theano.ifelse import ifelse
import sys
import argparse
import random
#from theano.tensor.shared_randomstreams import RandomStreams 
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
#faster random generator
import optimization
import util
from theano.tensor.nnet import conv2d
#from theano.compile.nanguardmode import NanGuardMode
try: # <= 0.8.2
    from theano.tensor.signal.downsample import max_pool_2d
except: # >= 0.9.0
    #https://github.com/Theano/Theano/issues/4337
    from theano.tensor.signal.pool import pool_2d
    max_pool_2d = pool_2d

logger = logging.getLogger(__name__)
mode = theano.Mode(linker='cvm')
srng = RandomStreams(seed=234)

class DeepBind():
    """
    The model simulating original DeepBind
    """
    def __init__(self, num_motif, max_motif_len, embed, L1_reg=0.00, L2_reg=0.00, gradient_clip = 1, param_clip = 1, optimizer = "sgd"):
        # The shape of the tensor is: [number of feature maps at layer m, number of feature maps at layer m-1, filter height, filter width]
        motif_filter_size = (num_motif, 4, max_motif_len, 1)
        motif_filter = util.createRandomShareAdaDelta(motif_filter_size, 'motif_filter', mean=0, std=0.01)

        rec_bias = util.createRandomShareAdaDelta((num_motif), 'rec_bias', const=-4)
        W_h = util.createRandomShareAdaDelta((num_motif, 32), 'W_h', mean=0, std=0.01)
        b_h = util.createRandomShareAdaDelta((32), 'b_h', mean=0, std=0.01)

        W_out  = util.createRandomShareAdaDelta((32, 1), 'W_out', mean=0, std=0.01)
        b_out  = util.createRandomShareAdaDelta((1), 'b_out', mean=0, std=0.01)

        self.parameters = [motif_filter, rec_bias, W_h, b_h, W_out, b_out]
        [motif_filter, rec_bias, W_h, b_h, W_out, b_out] = [each[0] for each in self.parameters]

        # The shape of the tensor is as follows: [mini-batch size, number of input feature maps, image height, image width].
        # number of input feature maps is 4, height should be sequence length and width should always be 1
        dnaseq = T.tensor4(name='dnaseq', dtype=theano.config.floatX)
        target = T.vector(name='target', dtype=theano.config.floatX)
        target_weight = T.vector(name='target_weight', dtype=theano.config.floatX)
        l_r = T.scalar('l_r')
        mom = T.scalar('mom')  

        predict = conv2d(dnaseq, motif_filter)
        predict = T.transpose(predict, (0,2,1,3))
        predict = predict.reshape((predict.shape[0], predict.shape[1], predict.shape[2]))
        predict = T.maximum(0, predict - rec_bias)
        predict = T.max(predict, axis=1)   # Now a [mini-batch size, num_motif] matrix

        saved1 = T.argmax(predict, axis=1);
        saved2 = predict[T.arange(saved1.shape[0]),saved1]
        saved1 = saved1.reshape([-1, 1])
        saved2 = saved2.reshape([-1, 1])
        saved = T.concatenate([saved1, saved2], axis=1)

        predict = T.maximum(0, T.dot(predict, W_h) + b_h)
        predict = T.nnet.sigmoid(T.dot(predict, W_out) + b_out)
        # Consecutive pairs are forward and backword sequences. 
        predict = predict.reshape((-1,2))
        predict = T.max(predict, axis=1)   
        loss = -target * T.log(predict + 1e-5) - (1-target) * T.log(1 - predict + 1e-5)
        loss = loss * target_weight
        loss = T.mean(loss)

        L1 = 0
        L2 = 0
        for param in self.parameters:
            L1 += abs(param[0].sum())
            L2 += (param[0] ** 2).sum()
        cost = loss + L1_reg * L1 + L2_reg * L2

        if (optimizer == "sgd"):
            parameters = [[each[0], each[1]] for each in self.parameters]
            updates = optimization.sgd(cost, parameters, mom, l_r, gradient_clip, param_clip)
            self.train_func = theano.function(inputs=[dnaseq, target, target_weight, l_r, mom],
                                                  outputs=[cost, predict],
                                                  updates=updates,
                                                  mode=mode)
        elif (optimizer == "adadelta"):
            updates = optimization.adadelta(cost, self.parameters, param_clip, l_r)
            self.train_func = theano.function(inputs=[dnaseq, target, target_weight, l_r],
                                                  outputs=[cost, predict],
                                                  updates=updates,
                                                  mode=mode)

        self.error_func = theano.function(inputs=[dnaseq, target, target_weight],
                                          outputs=[cost, predict],
                                          mode=mode)

        self.debug_func = theano.function(inputs=[dnaseq],
                                          outputs=saved,
                                          mode=mode)

        self.predict_func = theano.function(inputs=[dnaseq],
                                          outputs=predict,
                                          mode=mode)

        self.optimizer = optimizer

class TFImputeModel():
    """    
    """
    def __init__(self, num_motif, max_motif_len, embed, L1_reg=0.00, L2_reg=0.00, gradient_clip = 1, param_clip = 1, optimizer = "sgd", seq_len = 300, tfcell_comb_cnt = 0):
        num_motif = num_motif.split(",")
        num_motif = [int(each) for each in num_motif]
        if (len(num_motif) != 3):
            sys.stderr.write("TFImputeModel require num_motif parameter to be X,Y,Z format.\n")
        hidden_size = num_motif[2]
        max_window = num_motif[1]
        num_motif = num_motif[0]
        window_cnt = seq_len // max_window
  
        full_product_count = 1
        embed_len = 0
        embed_sum = 0
        startidx = [0]
        for each in embed:
            embed_len += each[1]
            embed_sum += each[0] * each[1]
            startidx.append(embed_sum)
            full_product_count *= each[0]

        if (tfcell_comb_cnt == 0):
            tfcell_comb_cnt = full_product_count

        self.tfcell_comb_cnt = tfcell_comb_cnt

        self.embed = [tuple(each) for each in embed]
        self.startidx = startidx

        embedding = util.createRandomShareAdaDelta((embed_sum), 'tf_embedding')

        # The shape of the tensor is: [number of feature maps at layer m, number of feature maps at layer m-1, filter height, filter width]
        motif_filter_size = (num_motif, 4, max_motif_len, 1)
        motif_filter = util.createRandomShareAdaDelta(motif_filter_size, 'motif_filter', mean=0, std=0.01)

        W_g = util.createRandomShareAdaDelta((embed_len, num_motif), 'W_g', mean=0, std=0.01)
        b_g = util.createRandomShareAdaDelta((num_motif), 'b_g', mean=0, std=0.01)

        rec_bias = util.createRandomShareAdaDelta((num_motif), 'rec_bias', const=-4)

        W_h = util.createRandomShareAdaDelta((window_cnt * num_motif, hidden_size), 'W_h', mean=0, std=0.01)
        b_h = util.createRandomShareAdaDelta((hidden_size), 'b_h', mean=0, std=0.01)

        # This shared variable is special. It will be useful only after prepare_batch_predict is called
        # In prepare_batch_predict, the values will be reset
        self.precompute_gate = util.createZeroShare(size=(tfcell_comb_cnt, num_motif), name='precompute_gate')

        self.parameters = [embedding, motif_filter, rec_bias, W_g, b_g, W_h, b_h]
        [embedding, motif_filter, rec_bias, W_g, b_g, W_h, b_h] = [each[0] for each in self.parameters]

        # The shape of the tensor is as follows: [mini-batch size, number of input feature maps, image height, image width].
        # number of input feature maps is 4, height should be sequence length and width should always be 1
        dnaseq = T.tensor4(name='dnaseq', dtype=theano.config.floatX)
        idxes = T.matrix(name='idxes', dtype='int32')
        target = T.vector(name='target', dtype=theano.config.floatX)
        target_weight = T.vector(name='target_weight', dtype=theano.config.floatX)
        l_r = T.scalar('l_r')
        mom = T.scalar('mom')  

        predict = conv2d(dnaseq, motif_filter)
        rec_bias = rec_bias.reshape((1, num_motif, 1, 1))
        predict = T.maximum(0, predict - rec_bias)

        predict = max_pool_2d(predict, (max_window, 1), ignore_border = True)
        predict = T.transpose(predict, (0,2,1,3))
        predict = predict.reshape((predict.shape[0], predict.shape[1], predict.shape[2]))
        # Now [batch_size, window_cnt, num_motif]
        seqfeature = predict

        em = []
        for i in range(len(embed)):
            sidx = startidx[i]
            eidx = startidx[i+1]
            curr = embedding[sidx: eidx]
            curr = curr.reshape(embed[i])
            em.append(curr[idxes[:,i]])

        embed = T.concatenate(em, axis=1)
        embed = embed.reshape((embed.shape[0], 1, embed.shape[1]))
        gate = T.nnet.sigmoid(T.dot(embed, W_g) + b_g)
        predict = seqfeature * gate
        predict = predict.reshape((predict.shape[0], -1))
        predict = T.dot(predict, W_h) + b_h
        predict = T.max(predict, axis=1)
        predict = T.nnet.sigmoid(predict)
        # Consecutive pairs are forward and backward sequences. 
        predict = predict.reshape((-1,2))
        predict = T.max(predict, axis=1)   
        loss = -target * T.log(predict + 1e-5) - (1-target) * T.log(1 - predict + 1e-5)
        loss = loss * target_weight
        loss = T.mean(loss)

        # The precompute_gate shape is: (tfcell_comb_cnt, num_motif)
        # The seqfeature shape is: (batch_size, window_cnt, num_motif)
        # The precompute_gate shape could be: (12376, 2000)
        # The seqfeature shape could be: (16, 3, 2000)
        precompute_gate = T.extra_ops.repeat(self.precompute_gate, seqfeature.shape[0] * window_cnt, axis=0)
        bat_pred = seqfeature.reshape((-1, num_motif))
        bat_pred = T.tile(bat_pred, (tfcell_comb_cnt, 1))
        bat_pred = bat_pred * precompute_gate  # shape: [tfcell_comb_cnt * batch_size * window_cnt, num_motif]
        bat_pred = bat_pred.reshape((tfcell_comb_cnt, seqfeature.shape[0] // 2, 2, -1))
        bat_pred = T.dot(bat_pred, W_h) + b_h
        bat_pred = T.max(bat_pred, axis=3)
        bat_pred = T.nnet.sigmoid(bat_pred)
        bat_pred = T.max(bat_pred, axis=2)   
        bat_pred = bat_pred.transpose([1,0])

        L1 = 0
        L2 = 0
        for param in [W_g]:
            L1 += T.mean(abs(param))
            L2 += T.mean(param ** 2)
        cost = loss + L1_reg * L1 + L2_reg * L2

        if (optimizer == "sgd"):
            parameters = [[each[0], each[1]] for each in self.parameters]
            updates = optimization.sgd(cost, parameters, mom, l_r, gradient_clip, param_clip)
            self.train_func = theano.function(inputs=[idxes, dnaseq, target, target_weight, l_r, mom],
                                                  outputs=[cost, predict],
                                                  updates=updates,
                                                  mode=mode)
        elif (optimizer == "adadelta"):
            updates = optimization.adadelta(cost, self.parameters, param_clip, l_r)
            self.train_func = theano.function(inputs=[idxes, dnaseq, target, target_weight, l_r],
                                                  outputs=[cost, predict],
                                                  updates=updates,
                                                  mode=mode)

        self.error_func = theano.function(inputs=[idxes, dnaseq, target, target_weight], outputs=[loss, predict], mode=mode)

        self.predict_func = theano.function(inputs=[idxes, dnaseq], outputs=predict, mode=mode)

        self.batch_predict_func = theano.function(inputs=[dnaseq], outputs=bat_pred, mode=mode)

        # After the model is trained,
        # given the DNA sequence, output the learned sequence feature.
        # The calculation is independent of TF and cell type
        seqfeature = seqfeature.reshape([seqfeature.shape[0], -1])
        self.seqfeature_func = theano.function(inputs=[dnaseq], outputs=seqfeature, mode=mode)

        # After the model is trained,
        # call this function to precalculate the gates
        gate = gate.reshape((gate.shape[0], -1))
        self.gate_func = theano.function(inputs=[idxes], outputs=gate, mode=mode)

        self.optimizer = optimizer

    def get_embedding(self):
        # Use device=cpu 
        embedding = self.parameters[0][0].eval()
        em = []
        for i in range(len(self.embed)):
            sidx = self.startidx[i]
            eidx = self.startidx[i+1]
            curr = embedding[sidx: eidx]
            curr = curr.reshape(self.embed[i])
            em.append(curr)
        return em

    def get_seq_feature(self, dnaseq):
        r = self.seqfeature_func(dnaseq)
        return r

    # After the model is trained or the parameters are loaded from trained model
    # call this function to prepare to do batch test. 
    # This function will calculate the gate and set the special parameter 'precompute_gate' 
    # The parameter 'embed' is the same parameter used to initialize the model
    # Suppose that there are only two features
    def prepare_batch_predict(self, tfcellidxes):
        embed = self.embed
        if (len(embed) != 2):
            sys.stderr.write('prepare_batch_predict only accept embeddings with two features\n')
            exit(1)

        if (len(tfcellidxes) == 0):
            idxes = []
            for i in range(embed[0][0]):
                for j in range(embed[1][0]):
                    idxes.append([i, j])
        else:
            idxes = tfcellidxes
        idxes = np.array(idxes, dtype='int32')
        if (len(idxes) != self.tfcell_comb_cnt):
            sys.stderr.write("Error: tfcell_comb_cnt does not match to the listed tf-cell combinations\n")
            exit(1)

        gate = self.gate_func(idxes)
        self.precompute_gate.set_value(gate)

    # Call this function after prepare_batch_predict is called.
    def batch_predict(self, dnaseq):
        return self.batch_predict_func(dnaseq)

    def train(self, idxes, dnaseq, target, target_weight, l_r, mom):
        if (self.optimizer == "sgd"):
            r = self.train_func(idxes, dnaseq, target, target_weight, l_r, mom)
        elif (self.optimizer == "adadelta"):
            r = self.train_func(idxes, dnaseq, target, target_weight, l_r)
        return r

    def error(self, idxes, dnaseq, target, target_weight):
        return self.error_func(idxes, dnaseq, target, target_weight)

    def predict(self, idxes, dnaseq):
        return self.predict_func(idxes, dnaseq)

    def get_state(self):
        state = []
        for par in self.parameters:
            state.append([each.get_value() for each in par])
        return state

    def set_state(self, values):
        idx = 0
        for par in self.parameters:
            idx2 = 0
            for each in par:
                each.set_value(values[idx][idx2])
                idx2 += 1
            idx += 1

    def print_human(self):
        mf = self.parameters[0][0].eval()
        for i in range(len(mf)):
            print("Motif ", i)
            for j in range(len(mf[i])):
                for k in range(len(mf[i,j])):
                    print('%10.6f' % mf[i,j,k,0],)
                print("")


class TFImputeModelRNN(TFImputeModel):
    """    
    It uses a RNN on top of CNN to capture the motif order
    """
    def __init__(self, num_motif, max_motif_len, embed, L1_reg=0.00, L2_reg=0.00, gradient_clip = 1, param_clip = 1, optimizer = "sgd", batch_size = 32):
        num_motif = num_motif.split(",")
        num_motif = [int(each) for each in num_motif]
        if (len(num_motif) != 3):
            sys.stderr.write("TFImputeModelRNN require num_motif parameter to be <#Motif><Maxpooling window><HiddenSize> format.\n")
            exit()

        n_hidden = num_motif[2]
        window_size = num_motif[1]
        num_motif = num_motif[0]

        # Note that embed length is ignored. Use the sigmoid(embedding) as gate directly

        target_cnt = embed[0][0]   # For TF
        embed_cnt  = embed[1][0]   # For cell line
        embedding = util.createRandomShareAdaDelta((embed_cnt, num_motif), 'embedding')

        # The shape of the tensor is: [number of feature maps at layer m, number of feature maps at layer m-1, filter height, filter width]
        motif_filter_size = (num_motif, 4, max_motif_len, 1)
        motif_filter = util.createRandomShareAdaDelta(motif_filter_size, 'motif_filter', mean=0, std=0.01)

        rec_bias = util.createRandomShareAdaDelta((num_motif), 'rec_bias', const=-4)

        # First half is memory and second half is the traiditional hidden
        hidden_states = util.createZeroShare((batch_size, n_hidden), 'hidden_states')
        bias = util.createRandomShareAdaDelta((n_hidden * 3), 'bias')
        W_i2h = util.createRandomShareAdaDelta((num_motif, n_hidden * 3), 'W_i2h')
        W_h2h = util.createRandomShareAdaDelta((n_hidden, n_hidden * 3), 'W_h2h')
        W_h2o = util.createRandomShareAdaDelta((n_hidden, target_cnt), 'W_h2o')
        b_o = util.createRandomShareAdaDelta((target_cnt), 'b_o')

        self.parameters = [embedding, motif_filter, rec_bias, bias, W_h2h, W_i2h, W_h2o, embedding, b_o]
        [embedding, motif_filter, rec_bias, bias, W_h2h, W_i2h, W_h2o, embedding, b_o] = [each[0] for each in self.parameters]

        # The shape of the tensor is as follows: [mini-batch size, number of input feature maps, image height, image width].
        # number of input feature maps is 4, height should be sequence length and width should always be 1
        dnaseq = T.tensor4(name='dnaseq', dtype=theano.config.floatX)
        idxes = T.matrix(name='idxes', dtype='int32')
        target = T.vector(name='target', dtype=theano.config.floatX)
        target_weight = T.vector(name='target_weight', dtype=theano.config.floatX)
        l_r = T.scalar('l_r')
        mom = T.scalar('mom')  

        predict = conv2d(dnaseq, motif_filter)
        predict = max_pool_2d(predict, (window_size, 1), ignore_border = True)
        predict = T.transpose(predict, (2,0,1,3))

        # Now a [seq, mini-batch size, num_motif] matrix
        predict = predict.reshape((predict.shape[0], predict.shape[1], predict.shape[2]))
        predict = T.maximum(0, predict - rec_bias)
        embed = T.nnet.sigmoid(embedding[idxes[:,1]])
        embed = embed.reshape((1, embed.shape[0], embed.shape[1]))
        predict = embed * predict

        def step(x_t, h_tm1):
            gate = T.dot(x_t, W_i2h[:,n_hidden:]) + T.dot(h_tm1, W_h2h[:,n_hidden:]) + bias[n_hidden:]
            g_r = T.nnet.sigmoid(gate[:,:n_hidden])
            g_u = T.nnet.sigmoid(gate[:,n_hidden:])
            h_t = T.tanh(T.dot(x_t, W_i2h[:,:n_hidden]) + T.dot(h_tm1 * g_r, W_h2h[:,:n_hidden]))
            h_t = (1 - g_u) * h_tm1 + g_u * h_t
            return h_t

        hidden_states = hidden_states[:predict.shape[1]]
        predict, _ = theano.scan(step, sequences=predict, outputs_info=[hidden_states])
        predict = T.nnet.sigmoid(T.dot(predict[-1], W_h2o) + b_o)

        predict = predict[T.arange(predict.shape[0]),idxes[:,0]]   # Take the corresponding output for current TF
        # Consecutive pairs are forward and backword sequences. 
        predict = predict.reshape((-1,2))
        predict = T.max(predict, axis=1)   
        loss = -target * T.log(predict + 1e-5) - (1-target) * T.log(1 - predict + 1e-5)
        loss = loss * target_weight
        loss = T.mean(loss)

        L1 = 0
        L2 = 0
        for param in self.parameters:
            L1 += abs(param[0].sum())
            L2 += (param[0] ** 2).sum()
        cost = loss + L1_reg * L1 + L2_reg * L2

        if (optimizer == "sgd"):
            parameters = [[each[0], each[1]] for each in self.parameters]
            updates = optimization.sgd(cost, parameters, mom, l_r, gradient_clip, param_clip)
            self.train_func = theano.function(inputs=[idxes, dnaseq, target, target_weight, l_r, mom],
                                                  outputs=[cost, predict],
                                                  updates=updates,
                                                  mode=mode)
        elif (optimizer == "adadelta"):
            updates = optimization.adadelta(cost, self.parameters, param_clip, l_r)
            self.train_func = theano.function(inputs=[idxes, dnaseq, target, target_weight, l_r],
                                                  outputs=[cost, predict],
                                                  updates=updates,
                                                  mode=mode)

        self.error_func = theano.function(inputs=[idxes, dnaseq, target, target_weight],
                                          outputs=[loss, predict],
                                          mode=mode)

        self.predict_func = theano.function(inputs=[idxes, dnaseq],
                                          outputs=predict,
                                          mode=mode)

        self.optimizer = optimizer

class TFImputeModelRNNMask(TFImputeModel):
    """    
    It uses a RNN on top of CNN to capture the motif order
    """
    def __init__(self, num_motif, max_motif_len, embed, L1_reg=0.00, L2_reg=0.00, gradient_clip = 1, param_clip = 1, optimizer = "sgd", batch_size = 32):
        num_motif = num_motif.split(",")
        num_motif = [int(each) for each in num_motif]
        if (len(num_motif) != 3):
            sys.stderr.write("TFImputeModelRNN require num_motif parameter to be <#Motif><Maxpooling window><HiddenSize> format.\n")
            exit()

        n_hidden = num_motif[2]
        window_size = num_motif[1]
        num_motif = num_motif[0]

        # Note that embed length is ignored. Use the sigmoid(embedding) as gate directly

        embed_cnt  = embed[0][0]         # Cellline
        target_cnt = embed[1][0]         # TFs
        embedding = util.createRandomShareAdaDelta((embed_cnt, num_motif), 'embedding')

        # The shape of the tensor is: [number of feature maps at layer m, number of feature maps at layer m-1, filter height, filter width]
        motif_filter_size = (num_motif, 4, max_motif_len, 1)
        motif_filter = util.createRandomShareAdaDelta(motif_filter_size, 'motif_filter', mean=0, std=0.01)

        rec_bias = util.createRandomShareAdaDelta((num_motif), 'rec_bias', const=-4)

        # First half is memory and second half is the traiditional hidden
        hidden_states = util.createZeroShare((batch_size, n_hidden), 'hidden_states')
        bias = util.createRandomShareAdaDelta((n_hidden * 3), 'bias')
        W_i2h = util.createRandomShareAdaDelta((num_motif, n_hidden * 3), 'W_i2h')
        W_h2h = util.createRandomShareAdaDelta((n_hidden, n_hidden * 3), 'W_h2h')
        W_h2o = util.createRandomShareAdaDelta((n_hidden, target_cnt), 'W_h2o')
        b_o = util.createRandomShareAdaDelta((target_cnt), 'b_o')

        self.parameters = [embedding, motif_filter, rec_bias, bias, W_h2h, W_i2h, W_h2o, embedding, b_o]
        [embedding, motif_filter, rec_bias, bias, W_h2h, W_i2h, W_h2o, embedding, b_o] = [each[0] for each in self.parameters]

        # The shape of the tensor is as follows: [mini-batch size, number of input feature maps, image height, image width].
        # number of input feature maps is 4, height should be sequence length and width should always be 1
        dnaseq = T.tensor4(name='dnaseq', dtype=theano.config.floatX)
        idxes = T.matrix(name='idxes', dtype='int32')
        target = T.matrix(name='target', dtype='int32')
        mask = T.matrix(name='mask', dtype='int32')
        l_r = T.scalar('l_r')
        mom = T.scalar('mom')  

        predict = conv2d(dnaseq, motif_filter)

        predict = max_pool_2d(predict, (window_size, 1), ignore_border = True)
        predict = T.transpose(predict, (2,0,1,3))

        # Now a [seq, mini-batch size, num_motif] matrix
        predict = predict.reshape((predict.shape[0], predict.shape[1], predict.shape[2]))
        predict = T.maximum(0, predict - rec_bias)
        embed = T.nnet.sigmoid(embedding[idxes[:,0]])
        embed = embed.reshape((1, embed.shape[0], embed.shape[1]))
        predict = embed * predict

        def step(x_t, h_tm1):
            gate = T.dot(x_t, W_i2h[:,n_hidden:]) + T.dot(h_tm1, W_h2h[:,n_hidden:]) + bias[n_hidden:]
            g_r = T.nnet.sigmoid(gate[:,:n_hidden])
            g_u = T.nnet.sigmoid(gate[:,n_hidden:])
            h_t = T.tanh(T.dot(x_t, W_i2h[:,:n_hidden]) + T.dot(h_tm1 * g_r, W_h2h[:,:n_hidden]))
            h_t = (1 - g_u) * h_tm1 + g_u * h_t
            return h_t

        hidden_states = hidden_states[:predict.shape[1]]
        predict, _ = theano.scan(step, sequences=predict, outputs_info=[hidden_states])
        predict = T.nnet.sigmoid(T.dot(predict[-1], W_h2o) + b_o)

        # Consecutive pairs are forward and backword sequences. 
        predict = predict.reshape((predict.shape[0] / 2, 2, -1))
        predict = T.max(predict, axis=1)   
        loss = -mask * target * T.log(predict + 1e-5) - mask * (1-target) * T.log(1 - predict + 1e-5)
        loss = T.sum(loss) / T.sum(mask)

        L1 = 0
        L2 = 0
        for param in self.parameters:
            L1 += abs(param[0].sum())
            L2 += (param[0] ** 2).sum()
        cost = loss + L1_reg * L1 + L2_reg * L2

        if (optimizer == "sgd"):
            parameters = [[each[0], each[1]] for each in self.parameters]
            updates = optimization.sgd(cost, parameters, mom, l_r, gradient_clip, param_clip)
            self.train_func = theano.function(inputs=[idxes, dnaseq, target, mask, l_r, mom],
                                                  outputs=[cost, predict],
                                                  updates=updates,
                                                  mode=mode)
        elif (optimizer == "adadelta"):
            updates = optimization.adadelta(cost, self.parameters, param_clip, l_r)
            self.train_func = theano.function(inputs=[idxes, dnaseq, target, mask, l_r],
                                                  outputs=[cost, predict],
                                                  updates=updates,
                                                  mode=mode)

        self.error_func = theano.function(inputs=[idxes, dnaseq, target, mask],
                                          outputs=[loss, predict],
                                          mode=mode)

        self.predict_func = theano.function(inputs=[idxes, dnaseq],
                                          outputs=predict,
                                          mode=mode)

        self.optimizer = optimizer
