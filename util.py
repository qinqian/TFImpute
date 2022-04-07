import numpy as np
import theano
import theano.tensor as T
import logging
import pickle

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s',
                    filemode='w')
logger = logging

def createRandomShare(size, name, low=None, high=None, mean=0, std=0.01, const=None):
    if low != None and high != None:
        dummy = np.asarray(np.random.uniform(size=size, low=low, high=high), dtype=theano.config.floatX)
    elif const != None:
        dummy = np.empty(size)
        dummy.fill(const, dtype=theano.config.floatX)
    elif mean != None and std != None:
        dummy = np.asarray(np.random.normal(mean, std, size=size), dtype=theano.config.floatX)
    else:
        print(name, low, high, mean, std, const)
        raise Exception("Internal Error: createRandomShare")
    share = theano.shared(value=dummy, name=name) 
    return share

def createZeroShare(size, name):
    dummy = np.asarray(np.zeros(size), dtype=theano.config.floatX)
    share = theano.shared(value=dummy, name=name) 
    return share

def getActivation(activation):
    if activation == 'tanh':
        activation = T.tanh
    elif activation == 'sigmoid':
        activation = T.nnet.sigmoid
    elif activation == 'relu':
        activation = lambda x: x * (x > 0)
    elif activation == 'cappedrelu':
        activation = lambda x: T.minimum(x * (x > 0), 6)
    else:
        print(activation)
        raise NotImplementedError
    return activation

def stable_sigmoid(x):
    x = T.exp(x - x.max(axis=1, keepdims=True))
    x = x / x.sum(axis=1, keepdims=True)
    return x

def soft3d(x):
    temp = T.reshape(x, (x.shape[0] * x.shape[1], -1))
    temp  = T.exp(temp - temp.max(axis=1, keepdims=True))
    temp = temp / temp.sum(axis=1, keepdims=True)
    x = T.reshape(temp, x.shape)
    return x

def createRandomShareAdaDelta(shape, name, low=-.01, high=.01, mean=None, std=None, const=None):
    parameter = createRandomShare(shape, name, low=low, high=high, mean=mean, std=std, const=const)
    parameter_update = createZeroShare(shape, name + "_updates")
    history_update = createZeroShare(shape, name + "_hist_updates")
    history_gradient = createZeroShare(shape, name + "_hist_grads")
    return parameter, parameter_update, history_update, history_gradient

def histgram(matrix, bins):
    hist = np.histogram(matrix, bins=20, density=True)
    hist = [(hist[0] / np.sum(hist[0])).tolist(), hist[1].tolist()]
    hist[0] = " ".join(['{0:.3}'.format(each) for each in hist[0]])
    hist[1] = " ".join(['{0:.3}'.format(each) for each in hist[1]])
    return hist

def matrix_stat(matrix, name):
    mi = np.amin(matrix)
    ma = np.amax(matrix)
    me = np.median(matrix)
    av = np.mean(matrix)
    std = np.std(matrix)
    hist = histgram(matrix, 20)
    return "%s mi=%f ma=%f me=%f av=%f std=%f hist=%s bin=%s" % (name, mi, ma, me, av, std, str(hist[0]), str(hist[1]))

class ProgressMonitor:
    def __init__(self, avg_window, logger, name, log_freq = 1):
        self.counter = 0
        self.sum = 0
        self.avg_window = avg_window
        self.since_last_sum = 0
        self.next_hit = avg_window
        if (self.next_hit == 0):
            self.next_hit = 1
        self.logger = logger
        self.name = name
        self.log_freq = log_freq
        self.log_counter = 0

    def progress(self, value):
        self.counter += 1
        self.sum += value
        self.since_last_sum += value
        if (self.counter == self.next_hit):
            if (self.avg_window == 0):
                added_counter = self.next_hit
                last_counter = added_counter / 2
            else:
                added_counter = self.avg_window
                last_counter = added_counter
            self.next_hit += added_counter
            if (last_counter == 0): last_counter = 1
            ret = [self.counter, self.sum / self.counter, self.since_last_sum / last_counter, value]
            self.since_last_sum = 0
            self.log_counter += 1
            if (self.logger and self.log_counter % self.log_freq == 0):
                self.logger.info('Counter: %s [%s] avg : %f, avg since last : %f, curr : %f' % 
                                 (str(self.counter).ljust(15), self.name, ret[1], ret[2], value))
            return ret
        return None

    def final(self):
        self.next_hit = self.counter
        self.counter -= 1
        return self.progress(0)

def report_parameter(config):
    logger.info("Parameters for current run")
    for key, value in config.__dict__.items():
        logger.info("    %s:  %s" % (str(key), str(value)))
    logger.info("")

def save(state, fname):
    """ Save a pickled representation of Model state. """
    file = open(fname, 'wb')
    pickle.dump(state, file, protocol=pickle.HIGHEST_PROTOCOL)
    file.close()

def load(path):
    """ Load model parameters from path. """
    logger.info("Loading parameters from %s ..." % path)
    file = open(path, 'rb')
    state = pickle.load(file)
    file.close()
    return state

