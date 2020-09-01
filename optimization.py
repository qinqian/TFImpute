""" V3. BPTT

@author Jianxing Feng
"""
import theano.tensor as T
from collections import OrderedDict
from theano.ifelse import ifelse


def sgd(cost, parameters, mom, l_r, gradient_clip, param_clip = 0, consider_constant = None):
    """ 
    Each element of parameters is an array with 4 elements:
        param, update, hist_grad, hist_update
    """
    updates_for_func = OrderedDict()
    for param, update in parameters:
        gparam = T.grad(cost, param, consider_constant = consider_constant)
        gparam = ifelse(T.isnan(T.sum(gparam)), T.zeros_like(gparam), gparam)

        upd = mom * update - l_r * gparam
        if (gradient_clip > 0):
            gradient_len = T.sqrt(T.sum(upd ** 2)) + 0.0000001   # To avoid zero divident
            upd = ifelse(T.lt(gradient_len, gradient_clip), 
                         upd,
                         upd / gradient_len * gradient_clip)
        updates_for_func[update] = upd

        new_weight = param + upd
        if (param_clip > 0):
            new_weight = T.clip(new_weight, -param_clip, param_clip)
        updates_for_func[param] = new_weight

    return updates_for_func

def adadelta(cost, parameters, param_clip = 0, l_r = 1.0, decay = 0.95, consider_constant = None):
    """ 
    Each element of parameters is an array with 4 elements:
        param, update, hist_grad, hist_update
    """
    updates_for_func = OrderedDict()
    for param, update, hist_grad, hist_update in parameters:
        gparam = T.grad(cost, param, consider_constant = consider_constant)
        gparam = ifelse(T.isnan(T.sum(gparam)), T.zeros_like(gparam), gparam)
        new_hist_grad = decay * hist_grad + (1-decay) * (gparam ** 2)
        new_update = -l_r * T.sqrt(hist_update + 1e-6) / T.sqrt(new_hist_grad + 1e-6) * gparam
        if (param_clip > 0):
            new_update = T.clip(new_update, -param_clip, param_clip)
        new_param = param + new_update
        new_hist_update = decay * hist_update + (1-decay) * (new_update ** 2)

        # Note that the order is important
        updates_for_func[hist_grad] = new_hist_grad
        updates_for_func[update] = new_update
        updates_for_func[param] = new_param
        updates_for_func[hist_update] = new_hist_update

    return updates_for_func

