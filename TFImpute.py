""" 

@author Jianxing Feng
"""
import numpy as np
import theano
import theano.tensor as T
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
import TFImputeModel
import util

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s',
                    filemode='w')
logger = logging

mode = theano.Mode(linker='cvm')

model_config = 0

class Model:
    def __init__(self):
        global model_config
        cleaned_embed = [each for each in model_config.embed if each[0] != 0]
        max_motif_len = int(model_config.max_motif_len)
        seq_len = model_config.seq_len + 2 * (max_motif_len - 1)

        # Assume the 3rd column is for cell type
        tfcell_comb = []
        if (model_config.bp_comb):
            tfcell_comb = [each.strip().split() for each in open(model_config.bp_comb)]
        self.tfcell_comb = tfcell_comb

        if (model_config.cnn == "TFImputeModel"):
            self.cnn = TFImputeModel.TFImputeModel(num_motif = model_config.num_motif, 
                                       max_motif_len = int(model_config.max_motif_len),
                                       embed = cleaned_embed,
                                       L1_reg = model_config.L1_reg, 
                                       L2_reg = model_config.L2_reg, 
                                       gradient_clip = 0, 
                                       param_clip = 0, 
                                       optimizer = model_config.optimizer,
                                       seq_len = seq_len,
                                       tfcell_comb_cnt = len(self.tfcell_comb))
        elif (model_config.cnn == "TFImputeModelRNN"):
            self.cnn = TFImputeModel.TFImputeModelRNN(num_motif = model_config.num_motif, 
                                       max_motif_len = int(model_config.max_motif_len),
                                       embed = cleaned_embed,
                                       L1_reg = model_config.L1_reg, 
                                       L2_reg = model_config.L2_reg, 
                                       gradient_clip = 0, 
                                       param_clip = 0, 
                                       optimizer = model_config.optimizer,
                                       batch_size = model_config.batch_size * 2)
        elif (model_config.cnn == "TFImputeModelRNNMask"):
            self.cnn = TFImputeModel.TFImputeModelRNNMask(num_motif = model_config.num_motif, 
                                       max_motif_len = int(model_config.max_motif_len),
                                       embed = cleaned_embed,
                                       L1_reg = model_config.L1_reg, 
                                       L2_reg = model_config.L2_reg, 
                                       gradient_clip = 0, 
                                       param_clip = 0, 
                                       optimizer = model_config.optimizer,
                                       batch_size = model_config.batch_size * 2)

        else:
            sys.stderr.write("Wrong model " + model_config.cnn + "\n")
            exit(1)

        self.token2idx = [{} for each in model_config.embed]
 
    def get_state(self):
        return [self.token2idx, self.cnn.get_state()]

    def set_state(self, state):
        self.token2idx = state[0]
        self.cnn.set_state(state[1])

    def seq2nparray(self, dnasequence, seq_len):
        if (model_config.train != None and model_config.min_len > 0):
            rand_len = random.randint(model_config.min_len, seq_len)
            border = (len(dnasequence) - rand_len) // 2
            if (border > 0):
                dnasequence = dnasequence[border: border + rand_len]
        if (len(dnasequence) > seq_len):
            border = (len(dnasequence) - seq_len) // 2
            dnasequence = dnasequence[border: border + seq_len]
        elif (len(dnasequence) < seq_len):
            border1 = (seq_len - len(dnasequence)) // 2
            border2 = seq_len - len(dnasequence) - border1
            dnasequence = 'N' * border1 + dnasequence + 'N' * border2

        ## pad with (max_motif_len-1) N to the border
        max_motif_len = int(model_config.max_motif_len)
        dnasequence = 'N' * (max_motif_len-1) + dnasequence + 'N' *  (max_motif_len-1)

        arr = [[0] * 4 for each in dnasequence]
        for i in range(len(dnasequence)):
            curr = dnasequence[i]
            idx = 0 
            if (curr == 'A'): idx = 0
            elif (curr == 'T'): idx = 1
            elif (curr == 'C'): idx = 2
            elif (curr == 'G'): idx = 3
            elif (curr != 'N'): 
                sys.stderr.write("Input error " + curr + " is encountered\n")
                exit(1)
            if (curr == 'N'): arr[i] = [0.25] * 4
            else: arr[i][idx] = 1

        return np.array(arr, dtype=theano.config.floatX)

    def complement(self, dnasequence):
        comp = ""
        for each in dnasequence:
            if (each == 'A'): comp += "T"
            if (each == 'T'): comp += "A"
            if (each == 'C'): comp += "G"
            if (each == 'G'): comp += "C"
            if (each == 'N'): comp += "N"
        return comp

    def removeN(self, seq):
        if ('N' not in seq):
            return seq
        base = 'ATCG'
        newseq = ""
        for b in seq:
            if (b != 'N'): newseq += b
            else: newseq += random.choice(base)
        return newseq

    def shuffleFile(self, datafile, buffsize = 100000):
        buf = []
        for each in open(datafile):
            if (len(buf) == buffsize):
                idx = random.randint(0, buffsize-1)
                yield buf[idx]
                buf[idx] = each.strip()
            else:
                buf.append(each.strip())
        random.shuffle(buf)
        for each in buf:
            yield each

    def scanDataBatch(self, datafile, batch_size, seq_len, shuffle_data = False):
        if (model_config.cnn == "TFImputeModelRNNMask"):
            for each in self.scanDataBatchNoShuffleMask(datafile, batch_size, seq_len):
                yield each
        else:
            for each in self.scanDataBatchNoShuffle(datafile, batch_size, seq_len):
                yield each

    def scanDataBatchNoShuffle(self, datafile, batch_size, seq_len):
        batch_size *= 2

        buffer = []
        buffer_target = []
        openf = open(datafile)

        eof = False
        while True:
            try:
                #line = openf.next()
                line = next(openf)
            except StopIteration:
                eof = True

            if (len(buffer) < batch_size and not eof):
                fields = line.strip().split("\t")
                if (len(fields) != len(model_config.embed) + 2):
                    logger.error("Format error for line %s" % line.strip())
                    continue

                idxes = []
                for i in range(len(model_config.embed)):
                    if (model_config.embed[i][0] == 0): continue
                    if (fields[i] not in self.token2idx[i]):
                        self.token2idx[i][fields[i]] = len(self.token2idx[i])
                    idxes.append(self.token2idx[i][fields[i]])

                weight = float(fields[len(model_config.embed)])
                seq = fields[len(model_config.embed)+1].upper()
                buffer.append([idxes, self.seq2nparray(seq, seq_len)])
                buffer.append([idxes, self.seq2nparray(self.complement(seq[::-1]), seq_len)])
                buffer_target.append(weight)

            if (len(buffer) == batch_size or (eof and len(buffer) > 0)):
                idxes = np.array([each[0] for each in buffer], dtype='int32')
                seq = np.dstack([each[1] for each in buffer])
                seq = np.transpose(seq, (2,1,0))
                seq = np.reshape(seq, seq.shape + (1,))
                target = np.array(buffer_target, dtype=theano.config.floatX)
                weight = np.array([1] * (len(buffer) // 2), dtype=theano.config.floatX)
                buffer = []
                buffer_target = []
                yield idxes, seq, target, weight

            if (eof): break

    def scanDataBatchNoShuffleMask(self, datafile, batch_size, seq_len):
        """
        The input format is: <CellLine><Seq><Target><Mask>
        """
        batch_size *= 2

        cleaned_embed = [each for each in model_config.embed if each[0] != 0]
        target_size = cleaned_embed[1][0]

        buffer = []
        buffer_target = []
        buffer_mask = []
        openf = open(datafile)

        eof = False
        while True:
            try:
                line = openf.next()
            except StopIteration:
                eof = True

            if (len(buffer) < batch_size and not eof):
                fields = line.strip().split("\t")
                if (len(fields) != len(model_config.embed) + 2):
                    logger.error("Format error for line %s" % line.strip())
                    continue

                # Always take only the first embed as cell line
                # The second embed is the target (TFs)
                idxes = []
                for i in range(len(model_config.embed)):
                    if (fields[i] not in self.token2idx[i]):
                        self.token2idx[i][fields[i]] = len(self.token2idx[i])
                    idxes.append(self.token2idx[i][fields[i]])
                    break

                # In case the number of target is not equal to the setted output size
                # This happens during test
                if (len(fields[2]) < target_size):
                    fields[2] = fields[2] + '0' * (target_size - len(fields[2]))
                if (len(fields[3]) < target_size):
                    fields[3] = fields[3] + '0' * (target_size - len(fields[3]))

                target = np.array([int(each) for each in fields[2]])
                mask = np.array([int(each) for each in fields[3]])
                seq = fields[1].upper()
                buffer.append([idxes, self.seq2nparray(seq, seq_len)])
                buffer.append([idxes, self.seq2nparray(self.complement(seq[::-1]), seq_len)])
                buffer_target.append(target)
                buffer_mask.append(mask)

            if (len(buffer) == batch_size or (eof and len(buffer) > 0)):
                idxes = np.array([each[0] for each in buffer], dtype='int32')
                seq = np.dstack([each[1] for each in buffer])
                seq = np.transpose(seq, (2,1,0))
                seq = np.reshape(seq, seq.shape + (1,))
                target = np.array(buffer_target, dtype='int32')
                mask = np.array(buffer_mask, dtype='int32')
                buffer = []
                buffer_target = []
                buffer_mask = []
                yield idxes, seq, target, mask 

            if (eof): break

    def fit(self, trainfile, validfile, validfile2, nepochs, batch_size, seq_len):
        logger.info('... training')
        model_save_frequency = model_config.model_save_frequency / batch_size

        epoch = 0
        sum_loss = 0;
        total_cnt = 1;
        module_size = 1;
        loss_monitor_window = util.ProgressMonitor(1000, None, 'loss_monitor_window')
        last_loss_window = 0
        while (epoch < nepochs):
            epoch = epoch + 1
            for idxes, seq, target, weightOrmask in self.scanDataBatch(trainfile, batch_size, seq_len, True):
                train_loss, predict = self.cnn.train(idxes, seq, target, weightOrmask, model_config.learning_rate, model_config.momentum)
                train_loss = np.mean(train_loss)
                sum_loss += train_loss
                loss_window = loss_monitor_window.progress(train_loss)
                if (loss_window):
                    last_loss_window = loss_window[2]

                print('\rXXXXXXXXXXXXXXXXXXXXXXX INFO epoch: %s mini-batch: %s avg loss: %f, last1000 lost: %f, curr loss: %f, lr: %.6f ' % (
                    str(epoch).ljust(5), str(total_cnt).ljust(15), sum_loss
                    / total_cnt, last_loss_window, train_loss,
                    model_config.learning_rate),)
                sys.stdout.flush()
                if (total_cnt % module_size == 0):
                    print("\r",)
                    sys.stdout.flush()
                    module_size *= 2
                    logger.info('epoch: %s mini-batch: %s avg loss: %f, last1000 lost: %f, curr loss: %f, lr: %.6f ' % 
                                (str(epoch).ljust(5), str(total_cnt).ljust(15), sum_loss / total_cnt, last_loss_window, train_loss, model_config.learning_rate))
                if (total_cnt % model_save_frequency == 0):
                    model_config.learning_rate *= 1-model_config.learning_rate_decay
                    if (model_config.model != None):
                        print("\r",)
                        sys.stdout.flush()
                        mf = model_config.model + "." + str(total_cnt / model_save_frequency)
                        logger.info("Save model to %s ..." % mf)
                        util.save([model_config, self.get_state()], mf)
                    
                    if validfile != None:
                        loss = [self.cnn.error(idxes, seq, target, weightOrmask) for idxes, seq, target, weightOrmask in self.scanDataBatch(validfile, batch_size, seq_len)]
                        valid_loss = np.mean([np.mean(each[0]) for each in loss])
                        sys.stdout.flush()
                        logger.info('epoch: %i, valid loss: %f, lr: %.6f %s' % (epoch, valid_loss, model_config.learning_rate, ' ' * 30))
 
                    if validfile2 != None:
                        loss = [self.cnn.error(idxes, seq, target, weightOrmask) for idxes, seq, target, weightOrmask in self.scanDataBatch(validfile2, batch_size, seq_len)]
                        valid_loss = np.mean([np.mean(each[0]) for each in loss])
                        sys.stdout.flush()
                        logger.info('epoch: %i, valid2 loss: %f, lr: %.6f %s' % (epoch, valid_loss, model_config.learning_rate, ' ' * 30))

                total_cnt += 1

    def test(self, testfile, prediction, batch_size, seq_len):
        logger.info('Testing ...')
        if (prediction):
            predout = open(prediction, 'w')

        sum_loss = 0
        module_size = 1
        total_cnt = 0

        for idxes, seq, target, weightOrmask in self.scanDataBatch(testfile, batch_size, seq_len):
            total_cnt += 1
            loss, predict = self.cnn.error(idxes, seq, target, weightOrmask)
            sum_loss += np.mean(loss)
            if (total_cnt % module_size == 0):
                module_size *= 2
                logger.info('mini-batch: %s avg loss: %f' % (str(total_cnt).ljust(15), sum_loss / total_cnt))

            if (prediction):
                if (model_config.cnn == "TFImputeModelRNNMask"):
                    for each in zip(target, predict):
                        # In this case, each element of target and predict is a vector
                        # The length of both vectors could be different
                        out1 = "".join([str(e)     for e in each[0]])
                        out2 = " ".join(["%.5f" % e for e in each[1]])
                        predout.write("%s\t%s\n" % (out1, out2))
                else:
                    for each in zip(target, predict):
                        predout.write("%d\t%.5f\n" % each)


        if (prediction):
            predout.close()

    def batch_predict(self, testfile, prediction, batch_size, seq_len):
        logger.info('Batch Predicting ...')
        if (prediction):
            predout = open(prediction, 'w')

        module_size = 1
        total_cnt = 0

        self.tfcell_comb = [(self.token2idx[0][each[0]], self.token2idx[2][each[1]]) for each in self.tfcell_comb]
        self.cnn.prepare_batch_predict(self.tfcell_comb)

        for idxes, seq, target, weight in self.scanDataBatch(testfile, batch_size, seq_len):
            total_cnt += 1
            predict = self.cnn.batch_predict(seq)
            if (total_cnt % module_size == 0):
                module_size *= 2
                logger.info('mini-batch: %s' % (str(total_cnt).ljust(15)))

            if (prediction):
                i = 0
                for row in predict:
                    predout.write("%.5f" % row[0])
                    for each in row[1:]:
                        predout.write("\t%.5f" % each)
                    predout.write("\n")
                    i += 1

        if (prediction):
            predout.close()

    def get_seq_feature(self, testfile, prediction, batch_size, seq_len):
        logger.info('Get Sequence Feature ...')
        if (prediction):
            predout = open(prediction, 'w')

        module_size = 1
        total_cnt = 0

        self.cnn.prepare_batch_predict()

        for idxes, seq, target, weight in self.scanDataBatch(testfile, batch_size, seq_len):
            total_cnt += 1
            predict = self.cnn.get_seq_feature(seq)
            if (total_cnt % module_size == 0):
                module_size *= 2
                logger.info('mini-batch: %s' % (str(total_cnt).ljust(15)))

            if (prediction):
                i = 0
                for row in predict:
                    first=True
                    for each in row:
                        if (first): predout.write("%.5f" % each)
                        else:       predout.write("\t%.5f" % each)
                        first = False

                    if (i % 2 == 1): predout.write("\n")
                    else:            predout.write("\t")
                    i += 1

        if (prediction):
            predout.close()

    def print_human(self):
        embedding = self.cnn.get_embedding()
        cnt = 0
        for i in range(len(model_config.embed)):
            if (model_config.embed[i][0] == 0): continue
            pairs = self.token2idx[i].items() 
            pairs.sort(key=lambda x: x[1])
            for key,idx in pairs:
                sys.stdout.write("Embed_%d\t%s" % (i, key))
                for j in range(len(embedding[cnt][idx])):
                    sys.stdout.write("\t%.6f" % embedding[cnt][idx][j])
                sys.stdout.write("\n")
            cnt += 1

def process_command_line(argv):
    """ Processing command line """

    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(description="""Version 0.1. cnn_bind: This script build a deep learning model to model the ChIPseq peak data
                                     Execute this script using: OMP_NUM_THREADS=2 THEANO_FLAGS='device=cpu,openmp=True,exception_verbosity=high' python TFImpute.py""")

    # Manual:
    parser.add_argument("-sf", dest="seq_feature", action='store_true', help="""For each in put sequence, get the sequence feature. For each sequence, the forward and backward feature
                                                                            are concatenated.""")
    parser.add_argument("-bp", dest="batch_predict", action='store_true', help="""After model is train, do batch prediction. In this model, the TF and cell types will be ignored
                                                                  in the input. All possible TF and cell types in the model will calculated. The output for each input sequence will
                                                                  be an array with #TF * #CellLine elements. Suppose the TF is the first effective embedding and cell type is the
                                                                  second effective embedding. The the output is: TF1-Cell1 TF1-Cell2 ... """)
    parser.add_argument("-bpcomb", dest="bp_comb", help="""During batch predict, if this parameter is set, only tf cell type combinations appearing in the file will be used""")
    parser.add_argument("-ph", dest="print_human", action='store_true', help="""Print the model""")
    parser.add_argument("-ct", dest="continue_train", action='store_true', help="Load the model and continue training.")
    parser.add_argument("-train", dest="train", default=None, help="""Data for training. The format is: <TF name> <Condition> <Sequence>""")
    parser.add_argument("-valid", dest="valid", default=None, help="""Data for validation. The format is the same as -train""")
    parser.add_argument("-valid2", dest="valid2", default=None, help="""Data for validation2. The format is the same as -train""")
    parser.add_argument("-test", dest="test", default=None, help="""Data for testing. For testing, -M, -b, -p, -w will overwrite the values in loaded model and -m will be set to 1.
                                                                  many other parameters for training will not be used.""")
    parser.add_argument("-M", dest="model", default=None, help="""The trained model. If -train is specified, the model will be trained from scratch and save to the specified
                                                                  path. If -train is not specified but -test is specified, the model will be loaded to run on the test data.
                                                                  In this case, all parameters specified in the command line will be ignored except -w.""")
    parser.add_argument("-F", dest="model_save_frequency", default=1000000, type=int, help="Save the model every <model_frequency> number of instances.")
    parser.add_argument("-O", dest="optimizer", default='adadelta', help="Optimization methods. Candiates are: 'sgd' and 'adadelta'.")
    parser.add_argument("-e", dest="nepochs", default=1000, type=int, help="The number of epochs to run")
    parser.add_argument("-p", dest="prediction", default=None, help="""The file for prediction output. It is effective only when -train is not set but -test is set. Note that
                                                                the odd number output (counting from 1) is the true score and the even number output is the score of shuffled background.""")
    parser.add_argument("-cnn", dest="cnn", default="Simple", help="The cnn model used. Could be 'TFImputeModel'")
    parser.add_argument("-IW", dest="ignore_weight", action='store_true', help="Whether to ignore the weight.")
    parser.add_argument("-m", dest="batch_size", default=10, type=int, help="The number of blocks for each batch")
    parser.add_argument("-seqLen", dest="seq_len", default=101, type=int, help="The length of each DNA sequence")
    parser.add_argument("-minLen", dest="min_len", default=0, type=int, help="""The min length of each DNA sequence. If it is > 0, then a random length in [min_len, seq_len] will be 
                                                                    generated for each instance. The omitted sequence will be filled with N.""")
    parser.add_argument("-mml", dest="max_motif_len", default=32, help="The max length of motif. If can also be a set of numbers separated by ,")
    parser.add_argument("-nm", dest="num_motif", default=256, help="The number of motifs. It might be multiple numbers separated by , to indicate the number of motifs for each layer.")
    parser.add_argument("-embed", dest="embed", default=None, help="""A set of number pairs separated by COMMA. Two number of each pair are separated by :
                                                                    For example  10:50,5:20,300:100 means that the first three column of the data are for embedding. 
                                                                    The first column contains 10 unique tokens and embed with length 50
                                                                    The first column contains 5  unique tokens and embed with length 20
                                                                    The first column contains 300 unique tokens and embed with length 100""")
    parser.add_argument("-l", dest="learning_rate", type=float, default=1, help="Learning rate")
    parser.add_argument("-d", dest="learning_rate_decay", type=float, default=0, help="Learning rate decay of every epoch")
    parser.add_argument("-mom", dest="momentum", type=float, default=0.9, help="Momentum")

    parser.add_argument("-L1", dest="L1_reg", type=float, default=0.0, help="The weight penalty for the L1 norm")
    parser.add_argument("-L2", dest="L2_reg", type=float, default=0.0, help="The weight penalty for the L2 norm")
    parser.add_argument("-c", dest="clip", default=1, type=float, help="Clip the parameter gradient to avoid gradient exploding. Set this parameter to 0 to skip clipping")
    parser.add_argument("-C", dest="param_clip", default=0, type=float, help="Clip all the parameters avoid parameter exploding. Set this parameter to 0 to skip clipping")

    return parser.parse_args(argv)

def main(argv=None):
    global model_config
    model_config = process_command_line(argv)
    if (model_config.optimizer != 'sgd' and model_config.optimizer != 'adadelta'):
        logger.error("Wrong parameters for -O.")
        exit(1)

    t0 = time.time()
    if (model_config.train != None):
        if (model_config.embed):
            em = model_config.embed.split(",")
            em = [each.split(":") for each in em]
            model_config.embed = [[int(col) for col in row] for row in em]
        else:
            model_config.embed = [[0,0]]

        model = Model()

        if (model_config.model != None and model_config.continue_train):
            state = util.load(model_config.model)
            new_model_config = state[0]
            new_model_config.batch_size = model_config.batch_size
            new_model_config.prediction = model_config.prediction
            new_model_config.model = model_config.model
            new_model_config.train = model_config.train
            new_model_config.valid = model_config.valid
            new_model_config.valid2 = model_config.valid2
            model_config = new_model_config
            model.set_state(state[1])
        util.report_parameter(model_config)

        model.fit(model_config.train, model_config.valid, model_config.valid2, model_config.nepochs, model_config.batch_size, model_config.seq_len)
        if (model_config.model != None):
            util.save([model_config, model.get_state()], model_config.model)
    elif (model_config.test != None and model_config.model != None):
        state = util.load(model_config.model)
        new_model_config = state[0]
        new_model_config.batch_size = model_config.batch_size
        new_model_config.prediction = model_config.prediction
        new_model_config.model = model_config.model
        new_model_config.test = model_config.test
        new_model_config.batch_predict = model_config.batch_predict
        new_model_config.bp_comb = model_config.bp_comb
        new_model_config.seq_feature = model_config.seq_feature
        new_model_config.min_len = 0
        model_config = new_model_config
        util.report_parameter(model_config)

        model = Model()
        model.set_state(state[1])
        if (model_config.batch_predict):
            model.batch_predict(model_config.test, model_config.prediction, model_config.batch_size, model_config.seq_len)
        elif (model_config.seq_feature):
            model.get_seq_feature(model_config.test, model_config.prediction, model_config.batch_size, model_config.seq_len)
        else:
            model.test(model_config.test, model_config.prediction, model_config.batch_size, model_config.seq_len)

    elif (model_config.print_human):
        state = util.load(model_config.model)
        new_model_config = state[0]
        new_model_config.model = model_config.model
        new_model_config.bp_comb = model_config.bp_comb
        model_config = new_model_config
        util.report_parameter(model_config)
        model = Model()
        model.set_state(state[1])
        model.print_human()
    else:
        logger.error("Either -train is needed or -test and -M is needed")

    logger.info("Elapsed time: %f" % (time.time() - t0))

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    status = main()
    sys.exit(status)
