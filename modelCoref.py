import numpy
import time
import sys
import subprocess
import os
import random
import cPickle
import copy

import theano
from theano import tensor as T
from collections import OrderedDict, defaultdict
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample
import theano.tensor.shared_randomstreams

#########################SOME UTILITIES########################


def randomMatrix(r, c, scale=0.2):
    #W_bound = numpy.sqrt(6. / (r + c))
    W_bound = 1.
    return scale * numpy.random.uniform(low=-W_bound, high=W_bound,\
                   size=(r, c)).astype(theano.config.floatX)

def ortho_weight(ndim):
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype(theano.config.floatX)

def _slice(_x, n, dim):
    return _x[:,n*dim:(n+1)*dim]

###############################################################

##########################Optimization function################

def adadelta(ips,cost,names,parameters,gradients,lr,norm_lim,rho=0.95,eps=1e-6):
    zipped_grads = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_grad'%k) for k, p in zip(names, parameters)]
    running_up2 = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_rup2'%k) for k, p in zip(names, parameters)]
    running_grads2 = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_rgrad2'%k) for k, p in zip(names, parameters)]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, gradients)]
    rg2up = [(rg2, rho * rg2 + (1. - rho) * (g ** 2)) for rg2, g in zip(running_grads2, gradients)] 
    f_grad_shared = theano.function(ips, cost, updates=zgup+rg2up, on_unused_input='ignore')

    updir = [-T.sqrt(ru2 + eps) / T.sqrt(rg2 + eps) * zg for zg, ru2, rg2 in zip(zipped_grads, running_up2, running_grads2)]
    ru2up = [(ru2, rho * ru2 + (1. - rho) * (ud ** 2)) for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(parameters, updir)]
    
    if norm_lim > 0:
        param_up = clipGradient(param_up, norm_lim, names)

    f_param_update = theano.function([lr], [], updates=ru2up+param_up, on_unused_input='ignore')

    return f_grad_shared, f_param_update

def sgd(ips,cost,names,parameters,gradients,lr,norm_lim):
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad'%k) for k, p in zip(names, parameters)]
    gsup = [(gs, g) for gs, g in zip(gshared, gradients)]

    f_grad_shared = theano.function(ips, cost, updates=gsup, on_unused_input='ignore')

    pup = [(p, p - lr * g) for p, g in zip(parameters, gshared)]
    
    if norm_lim > 0:
        pup = clipGradient(pup, norm_lim, names)
    
    f_param_update = theano.function([lr], [], updates=pup, on_unused_input='ignore')

    return f_grad_shared, f_param_update

def clipGradient(updates, norm, names):
    id = -1
    res = []
    for p, g in updates:
        id += 1
        if not names[id].startswith('word') and 'multi' not in names[id] and p.get_value(borrow=True).ndim == 2:
            col_norms = T.sqrt(T.sum(T.sqr(g), axis=0))
            desired_norms = T.clip(col_norms, 0, T.sqrt(norm))
            scale = desired_norms / (1e-7 + col_norms)
            g = g * scale
            
        res += [(p, g)]
    return res          

###############################################################

def _dropout_from_layer(rng, layers, p):
    srng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(999999))
    # p=1-p because 1's indicate keep and p is prob of dropping
    res = []
    for layer in layers:
        mask = srng.binomial(n=1, p=1-p, size=layer.shape)
        # The cast is important because
        # int * float32 = float64 which pulls things off the gpu
        output = layer * T.cast(mask, theano.config.floatX)
        res += [output]
    return res

###############################Models###############################

def getConcatenation(embDict, vars, features, features_dim, tranpose=False):
    xs = []

    for ed in features:
        if features[ed] == 0:
            var = vars[ed] if not tranpose else vars[ed].T
            xs += [embDict[ed][T.cast(var.flatten(), dtype='int32')].reshape((var.shape[0], var.shape[1], features_dim[ed]))]
        elif features[ed] == 1:
            if not tranpose:
                xs += [vars[ed]]
            else:
                xs += [vars[ed].dimshuffle(1,0,2)]

    if len(xs) == 1:
        basex = xs[0]
    else:
        basex = T.cast(T.concatenate(xs, axis=2), dtype=theano.config.floatX)

    return basex

def getInverseConcatenation(embDict, vars, features, features_dim):
        
    ixs = []

    for ed in features:
        if features[ed] == 0:
            var = vars[ed].T[::-1]
            ixs += [embDict[ed][T.cast(var.flatten(), dtype='int32')].reshape((var.shape[0], var.shape[1], features_dim[ed]))]
        elif features[ed] == 1:
            ixs += [vars[ed].dimshuffle(1,0,2)[::-1]]                

    if len(ixs) == 1:
        ibasex = ixs[0]
    else:
        ibasex = T.cast(T.concatenate(ixs, axis=2), dtype=theano.config.floatX)
    
    return ibasex

def createMatrix(random, kGivens, name):
    if name in kGivens:
        if kGivens[name].shape == random.shape:
            print '<------ Using given ', name, '------>'
            return kGivens[name]
        else: print '<------', name, ' is a given knowledge but mismatch dimension ', kGivens[name].shape, random.shape, '------>'
    return random

def rnn_ff(inps, dim, hidden, batSize, prefix, params, names, kGivens={}):
    Wx  = theano.shared( createMatrix(randomMatrix(dim, hidden), kGivens, prefix + '_Wx') )
    Wh  = theano.shared( createMatrix(randomMatrix(hidden, hidden), kGivens, prefix + '_Wh') )
    bh  = theano.shared( createMatrix(numpy.zeros(hidden, dtype=theano.config.floatX), kGivens, prefix + '_bh') )
    #model.container['bi_h0']  = theano.shared(numpy.zeros(model.container['nh'], dtype=theano.config.floatX))

    # bundle
    params += [ Wx, Wh, bh ] #, model.container['bi_h0']
    names += [ prefix + '_Wx', prefix + '_Wh', prefix + '_bh' ] #, 'bi_h0'

    def recurrence(x_t, h_tm1):
        h_t = T.nnet.sigmoid(T.dot(x_t, Wx) + T.dot(h_tm1, Wh) + bh)
        return h_t

    h, _  = theano.scan(fn=recurrence, \
            sequences=inps, outputs_info=[T.alloc(0., batSize, hidden)], n_steps=inps.shape[0])
    
    return h
    
def rnn_gru(inps, dim, hidden, batSize, prefix, params, names, kGivens={}):
    Wc = theano.shared( createMatrix(numpy.concatenate([randomMatrix(dim, hidden), randomMatrix(dim, hidden)], axis=1), kGivens, prefix + '_Wc') )

    bc = theano.shared( createMatrix(numpy.zeros(2 * hidden, dtype=theano.config.floatX), kGivens, prefix + '_bc') )

    U = theano.shared( createMatrix(numpy.concatenate([ortho_weight(hidden), ortho_weight(hidden)], axis=1), kGivens, prefix + '_U') )
    Wx = theano.shared( createMatrix(randomMatrix(dim, hidden), kGivens, prefix + '_Wx') )

    Ux = theano.shared( createMatrix(ortho_weight(hidden), kGivens, prefix + '_Ux') )

    bx = theano.shared( createMatrix(numpy.zeros(hidden, dtype=theano.config.floatX), kGivens, prefix + '_bx') )

    #model.container['bi_h0'] = theano.shared(numpy.zeros(model.container['nh'], dtype=theano.config.floatX))

    # bundle
    params += [ Wc, bc, U, Wx, Ux, bx ] #, model.container['bi_h0']
    names += [ prefix + '_Wc', prefix + '_bc', prefix + '_U', prefix + '_Wx', prefix + '_Ux', prefix + '_bx' ] #, 'bi_h0'
    
    def recurrence(x_t, h_tm1):
        preact = T.dot(h_tm1, U)
        preact += T.dot(x_t, Wc) + bc

        r_t = T.nnet.sigmoid(_slice(preact, 0, hidden))
        u_t = T.nnet.sigmoid(_slice(preact, 1, hidden))

        preactx = T.dot(h_tm1, Ux)
        preactx = preactx * r_t
        preactx = preactx + T.dot(x_t, Wx) + bx

        h_t = T.tanh(preactx)

        h_t = u_t * h_tm1 + (1. - u_t) * h_t

        return h_t

    h, _  = theano.scan(fn=recurrence, \
            sequences=inps, outputs_info=[T.alloc(0., batSize, hidden)], n_steps=inps.shape[0])
    
    return h
    
def ffBidirectCore(inps, iinps, dim, hidden, batSize, prefix, iprefix, params, names, kGivens={}):

    bi_h = rnn_ff(inps, dim, hidden, batSize, prefix, params, names, kGivens=kGivens)
    
    ibi_h = rnn_ff(iinps, dim, hidden, batSize, iprefix, params, names, kGivens=kGivens)

    _ibi_h = ibi_h[::-1]
    
    bi_rep = T.cast(T.concatenate([ bi_h, _ibi_h ], axis=2).dimshuffle(1,0,2), dtype=theano.config.floatX)

    return bi_rep
    
def gruBidirectCore(inps, iinps, dim, hidden, batSize, prefix, iprefix, params, names, kGivens={}):

    bi_h = rnn_gru(inps, dim, hidden, batSize, prefix, params, names, kGivens=kGivens)
    
    ibi_h = rnn_gru(iinps, dim, hidden, batSize, iprefix, params, names, kGivens=kGivens)

    _ibi_h = ibi_h[::-1]

    bi_rep = T.cast(T.concatenate([ bi_h, _ibi_h ], axis=2).dimshuffle(1,0,2), dtype=theano.config.floatX)

    return bi_rep

def ffForward(embDict, vars, features, features_dim, dimIn, hidden, batch, prefix, params, names, kGivens={}):
    ix = getConcatenation(embDict, vars, features, features_dim, tranpose=True)
    
    i_h = rnn_ff(ix, dimIn, hidden, batch, prefix, params, names, kGivens=kGivens)
    
    rep = T.cast(i_h.dimshuffle(1,0,2), dtype=theano.config.floatX)
    
    return rep

def ffBackward(embDict, vars, features, features_dim, dimIn, hidden, batch, iprefix, params, names, kGivens={}):
    iix = getInverseConcatenation(embDict, vars, features, features_dim)
    
    ii_h = rnn_ff(iix, dimIn, hidden, batch, iprefix, params, names, kGivens=kGivens)
    
    _ii_h = ii_h[::-1]
    
    rep = T.cast(_ii_h.dimshuffle(1,0,2), dtype=theano.config.floatX)
    
    return rep

def ffBiDirect(embDict, vars, features, features_dim, dimIn, hidden, batch, prefix, params, names, kGivens={}):
    bix = getConcatenation(embDict, vars, features, features_dim, tranpose=True)
    ibix = getInverseConcatenation(embDict, vars, features, features_dim)
    
    return ffBidirectCore(bix, ibix, dimIn, hidden, batch, prefix + '_ffbi', prefix + '_ffibi', params, names, kGivens=kGivens)
    
def gruForward(embDict, vars, features, features_dim, dimIn, hidden, batch, prefix, params, names, kGivens={}):
    ix = getConcatenation(embDict, vars, features, features_dim, tranpose=True)
    
    i_h = rnn_gru(ix, dimIn, hidden, batch, prefix, params, names, kGivens=kGivens)
    
    rep = T.cast(i_h.dimshuffle(1,0,2), dtype=theano.config.floatX)
    
    return rep

def gruBackward(embDict, vars, features, features_dim, dimIn, hidden, batch, iprefix, params, names, kGivens={}):
    iix = getInverseConcatenation(embDict, vars, features, features_dim)
    
    ii_h = rnn_gru(iix, dimIn, hidden, batch, iprefix, params, names, kGivens=kGivens)
    
    _ii_h = ii_h[::-1]
    
    rep = T.cast(_ii_h.dimshuffle(1,0,2), dtype=theano.config.floatX)
    
    return rep

def gruBiDirect(embDict, vars, features, features_dim, dimIn, hidden, batch, prefix, params, names, kGivens={}):
    bix = getConcatenation(embDict, vars, features, features_dim, tranpose=True)
    ibix = getInverseConcatenation(embDict, vars, features, features_dim)
    
    return gruBidirectCore(bix, ibix, dimIn, hidden, batch, prefix + '_grubi', prefix + '_gruibi', params, names, kGivens=kGivens)
    
###############################CONVOLUTIONAL CONTEXT####################################

def convolutionalLayer(inpu, feature_map, batch, length, window, dim, prefix, params, names, kGivens={}):
    down = window / 2
    up = window - down - 1
    zodown = T.zeros((batch, 1, down, dim), dtype=theano.config.floatX)
    zoup = T.zeros((batch, 1, up, dim), dtype=theano.config.floatX)
    
    inps = T.cast(T.concatenate([zoup, inpu, zodown], axis=2), dtype=theano.config.floatX)
    
    fan_in = window * dim
    fan_out = feature_map * window * dim / length #(length - window + 1)

    filter_shape = (feature_map, 1, window, dim)
    image_shape = (batch, 1, length + down + up, dim)

    #if non_linear=="none" or non_linear=="relu":
    #    conv_W = theano.shared(0.2 * numpy.random.uniform(low=-1.0,high=1.0,\
    #                            size=filter_shape).astype(theano.config.floatX))
        
    #else:
    #    W_bound = numpy.sqrt(6. / (fan_in + fan_out))
    #    conv_W = theano.shared(numpy.random.uniform(low=-W_bound,high=W_bound,\
    #                            size=filter_shape).astype(theano.config.floatX))

    W_bound = numpy.sqrt(6. / (fan_in + fan_out))
    conv_W = theano.shared( createMatrix(numpy.random.uniform(low=-W_bound,high=W_bound, size=filter_shape).astype(theano.config.floatX), kGivens, prefix + '_convL_W_' + str(window)) )

    conv_b = theano.shared( createMatrix(numpy.zeros(filter_shape[0], dtype=theano.config.floatX), kGivens, prefix + '_convL_b_' + str(window)) )

    # bundle
    params += [ conv_W, conv_b ]
    names += [ prefix + '_convL_W_' + str(window), prefix + '_convL_b_' + str(window) ]

    conv_out = conv.conv2d(input=inps, filters=conv_W, filter_shape=filter_shape, image_shape=image_shape)

    conv_out = T.tanh(conv_out + conv_b.dimshuffle('x', 0, 'x', 'x'))

    return conv_out.dimshuffle(0,2,1,3).flatten(3)
    
def convContextLs(inps, feature_map, convWins, batch, length, dim, prefix, params, names, kGivens={}):
    cx = T.cast(inps.reshape((inps.shape[0], 1, inps.shape[1], inps.shape[2])), dtype=theano.config.floatX)

    fts = []
    for i, convWin in enumerate(convWins):
        fti = convolutionalLayer(cx, feature_map, batch, length, convWin, dim, prefix + '_winL' + str(i), params, names, kGivens=kGivens)
        fts += [fti]

    convRep = T.cast(T.concatenate(fts, axis=2), dtype=theano.config.floatX)

    return convRep
    
def LeNetConvPoolLayer(inps, feature_map, batch, length, window, dim, prefix, params, names, kGivens={}):
    fan_in = window * dim
    fan_out = feature_map * window * dim / (length - window + 1)

    filter_shape = (feature_map, 1, window, dim)
    image_shape = (batch, 1, length, dim)
    pool_size = (length - window + 1, 1)

    #if non_linear=="none" or non_linear=="relu":
    #    conv_W = theano.shared(0.2 * numpy.random.uniform(low=-1.0,high=1.0,\
    #                            size=filter_shape).astype(theano.config.floatX))
        
    #else:
    #    W_bound = numpy.sqrt(6. / (fan_in + fan_out))
    #    conv_W = theano.shared(numpy.random.uniform(low=-W_bound,high=W_bound,\
    #                            size=filter_shape).astype(theano.config.floatX))

    W_bound = numpy.sqrt(6. / (fan_in + fan_out))
    conv_W = theano.shared( createMatrix(numpy.random.uniform(low=-W_bound,high=W_bound, size=filter_shape).astype(theano.config.floatX), kGivens, prefix + '_conv_W_' + str(window)) )

    conv_b = theano.shared( createMatrix(numpy.zeros(filter_shape[0], dtype=theano.config.floatX), kGivens, prefix + '_conv_b_' + str(window)) )

    # bundle
    params += [ conv_W, conv_b ]
    names += [ prefix + '_conv_W_' + str(window), prefix + '_conv_b_' + str(window) ]

    conv_out = conv.conv2d(input=inps, filters=conv_W, filter_shape=filter_shape, image_shape=image_shape)

        
    conv_out_act = T.tanh(conv_out + conv_b.dimshuffle('x', 0, 'x', 'x'))
    conv_output = downsample.max_pool_2d(input=conv_out_act, ds=pool_size, ignore_border=True)

    return conv_output.flatten(2)

def convContext(inps, feature_map, convWins, batch, length, dim, prefix, params, names, kGivens={}):

    cx = T.cast(inps.reshape((inps.shape[0], 1, inps.shape[1], inps.shape[2])), dtype=theano.config.floatX)

    fts = []
    for i, convWin in enumerate(convWins):
        fti = LeNetConvPoolLayer(cx, feature_map, batch, length, convWin, dim, prefix + '_win' + str(i), params, names, kGivens=kGivens)
        fts += [fti]

    convRep = T.cast(T.concatenate(fts, axis=1), dtype=theano.config.floatX)

    return convRep
    
def nonConsecutiveConvLayer2(inpu, feature_map, batch, length, dim, prefix, params, names, kGivens={}):
    window = 2
    fan_in = window * dim
    fan_out = feature_map * window * dim / length #(length - window + 1)
    W_bound = numpy.sqrt(6. / (fan_in + fan_out))
    Ws = []
    for i in range(window):
        conv_W = theano.shared( createMatrix(numpy.random.uniform(low=-W_bound,high=W_bound, size=(dim, feature_map)).astype(theano.config.floatX), kGivens, prefix + '_convL_W_' + str(window) + '_' + str(i)) )
        
        Ws += [conv_W]
        
        params += [ conv_W ]
        names += [ prefix + '_convL_W_' + str(window) + '_' + str(i) ]
    
    conv_b = theano.shared( createMatrix(numpy.zeros(feature_map, dtype=theano.config.floatX), kGivens, prefix + '_convL_b_' + str(window)) )
    params += [ conv_b ]
    names += [ prefix + '_convL_b_' + str(window) ]
    
    def recurrence(_x, i_m1, i_m2):
        ati = T.dot(_x, Ws[0])
        _m1 = T.maximum(i_m1, ati)
        ati = i_m1 + T.dot(_x, Ws[1])
        _m2 = T.maximum(i_m2, ati)
        
        return [_m1, _m2]
    
    ms, _ = theano.scan(fn=recurrence, sequences=[inpu], outputs_info=[T.alloc(0., batch, feature_map), T.alloc(0., batch, feature_map)], n_steps=inpu.shape[0])
    
    res = T.tanh(ms[1][-1] + conv_b[numpy.newaxis,:])
    return res
    
def nonConsecutiveConvLayer3(inpu, feature_map, batch, length, dim, prefix, params, names, kGivens={}):
    window = 3
    fan_in = window * dim
    fan_out = feature_map * window * dim / length #(length - window + 1)
    W_bound = numpy.sqrt(6. / (fan_in + fan_out))
    Ws = []
    for i in range(window):
        conv_W = theano.shared( createMatrix(numpy.random.uniform(low=-W_bound,high=W_bound, size=(dim, feature_map)).astype(theano.config.floatX), kGivens, prefix + '_convL_W_' + str(window) + '_' + str(i)) )
        
        Ws += [conv_W]
        
        params += [ conv_W ]
        names += [ prefix + '_convL_W_' + str(window) + '_' + str(i) ]
    
    conv_b = theano.shared( createMatrix(numpy.zeros(feature_map, dtype=theano.config.floatX), kGivens, prefix + '_convL_b_' + str(window)) )
    params += [ conv_b ]
    names += [ prefix + '_convL_b_' + str(window) ]
    
    def recurrence(_x, i_m1, i_m2, i_m3):
        ati = T.dot(_x, Ws[0])
        _m1 = T.maximum(i_m1, ati)
        ati = i_m1 + T.dot(_x, Ws[1])
        _m2 = T.maximum(i_m2, ati)
        ati = i_m2 + T.dot(_x, Ws[2])
        _m3 = T.maximum(i_m3, ati)
        
        return [_m1, _m2, _m3]
    
    ms, _ = theano.scan(fn=recurrence, sequences=[inpu], outputs_info=[T.alloc(0., batch, feature_map), T.alloc(0., batch, feature_map), T.alloc(0., batch, feature_map)], n_steps=inpu.shape[0])
    
    res = T.tanh(ms[2][-1] + conv_b[numpy.newaxis,:])
    return res
    
def nonConsecutiveConvLayer4(inpu, feature_map, batch, length, dim, prefix, params, names, kGivens={}):
    window = 4
    fan_in = window * dim
    fan_out = feature_map * window * dim / length #(length - window + 1)
    W_bound = numpy.sqrt(6. / (fan_in + fan_out))
    Ws = []
    for i in range(window):
        conv_W = theano.shared( createMatrix(numpy.random.uniform(low=-W_bound,high=W_bound, size=(dim, feature_map)).astype(theano.config.floatX), kGivens, prefix + '_convL_W_' + str(window) + '_' + str(i)) )
        
        Ws += [conv_W]
        
        params += [ conv_W ]
        names += [ prefix + '_convL_W_' + str(window) + '_' + str(i) ]
    
    conv_b = theano.shared( createMatrix(numpy.zeros(feature_map, dtype=theano.config.floatX), kGivens, prefix + '_convL_b_' + str(window)) )
    params += [ conv_b ]
    names += [ prefix + '_convL_b_' + str(window) ]
    
    def recurrence(_x, i_m1, i_m2, i_m3, i_m4):
        ati = T.dot(_x, Ws[0])
        _m1 = T.maximum(i_m1, ati)
        ati = i_m1 + T.dot(_x, Ws[1])
        _m2 = T.maximum(i_m2, ati)
        ati = i_m2 + T.dot(_x, Ws[2])
        _m3 = T.maximum(i_m3, ati)
        ati = i_m3 + T.dot(_x, Ws[3])
        _m4 = T.maximum(i_m4, ati)
        
        return [_m1, _m2, _m3, _m4]
    
    ms, _ = theano.scan(fn=recurrence, sequences=[inpu], outputs_info=[T.alloc(0., batch, feature_map), T.alloc(0., batch, feature_map), T.alloc(0., batch, feature_map), T.alloc(0., batch, feature_map)], n_steps=inpu.shape[0])
    
    res = T.tanh(ms[3][-1] + conv_b[numpy.newaxis,:])
    return res
    
def nonConsecutiveConvLayer5(inpu, feature_map, batch, length, dim, prefix, params, names, kGivens={}):
    window = 5
    fan_in = window * dim
    fan_out = feature_map * window * dim / length #(length - window + 1)
    W_bound = numpy.sqrt(6. / (fan_in + fan_out))
    Ws = []
    for i in range(window):
        conv_W = theano.shared( createMatrix(numpy.random.uniform(low=-W_bound,high=W_bound, size=(dim, feature_map)).astype(theano.config.floatX), kGivens, prefix + '_convL_W_' + str(window) + '_' + str(i)) )
        
        Ws += [conv_W]
        
        params += [ conv_W ]
        names += [ prefix + '_convL_W_' + str(window) + '_' + str(i) ]
    
    conv_b = theano.shared( createMatrix(numpy.zeros(feature_map, dtype=theano.config.floatX), kGivens, prefix + '_convL_b_' + str(window)) )
    params += [ conv_b ]
    names += [ prefix + '_convL_b_' + str(window) ]
    
    def recurrence(_x, i_m1, i_m2, i_m3, i_m4, i_m5):
        ati = T.dot(_x, Ws[0])
        _m1 = T.maximum(i_m1, ati)
        ati = i_m1 + T.dot(_x, Ws[1])
        _m2 = T.maximum(i_m2, ati)
        ati = i_m2 + T.dot(_x, Ws[2])
        _m3 = T.maximum(i_m3, ati)
        ati = i_m3 + T.dot(_x, Ws[3])
        _m4 = T.maximum(i_m4, ati)
        ati = i_m4 + T.dot(_x, Ws[4])
        _m5 = T.maximum(i_m5, ati)
        
        return [_m1, _m2, _m3, _m4, _m5]
    
    ms, _ = theano.scan(fn=recurrence, sequences=[inpu], outputs_info=[T.alloc(0., batch, feature_map), T.alloc(0., batch, feature_map), T.alloc(0., batch, feature_map), T.alloc(0., batch, feature_map), T.alloc(0., batch, feature_map)], n_steps=inpu.shape[0])
    
    res = T.tanh(ms[4][-1] + conv_b[numpy.newaxis,:])
    return res
    
def nonConsecutiveConvNet(inps, feature_map, convWins, batch, length, dim, prefix, params, names, kGivens={}):

    cx = inps.dimshuffle(1,0,2)

    fts = []
    for i, convWin in enumerate(convWins):
        fti = eval('nonConsecutiveConvLayer' + str(convWin))(cx, feature_map, batch, length, dim, prefix + '_nonCons_win' + str(i), params, names, kGivens=kGivens)
        fts += [fti]

    convRep = T.cast(T.concatenate(fts, axis=1), dtype=theano.config.floatX)

    return convRep
    
#############################Multilayer NNs################################

def HiddenLayer(inputs, nin, nout, params, names, prefix, kGivens={}):
    W_bound = numpy.sqrt(6. / (nin + nout))
    multi_W = theano.shared( createMatrix(numpy.random.uniform(low=-W_bound,high=W_bound, size=(nin, nout)).astype(theano.config.floatX), kGivens, prefix + '_multi_W') )

    multi_b = theano.shared( createMatrix(numpy.zeros(nout, dtype=theano.config.floatX), kGivens, prefix + '_multi_b') )
    res = []
    for input in inputs:
        out = T.nnet.sigmoid(T.dot(input, multi_W) + multi_b)
        res += [out]
    
    params += [multi_W, multi_b]
    names += [prefix + '_multi_W', prefix + '_multi_b']
    
    return res

def MultiHiddenLayers(inputs, hids, params, names, prefix, kGivens={}):
    
    hiddenVector = inputs
    id = 0
    for nin, nout in zip(hids, hids[1:]):
        id += 1
        hiddenVector = HiddenLayer(hiddenVector, nin, nout, params, names, prefix + '_layer' + str(id), kGivens=kGivens)
    return hiddenVector

#########################################################################################

class BaseModel(object):

    def __init__(self, args):
        '''
        nh :: dimension of the hidden layer
        nc :: number of classes
        ne :: number of word embeddings in the vocabulary
        #de :: dimension of the word embeddings
        cs :: word window context size
        '''
        self.container = {}
        
        self.args = args
        self.args['rng'] = numpy.random.RandomState(3435)
        self.args['dropout'] = args['dropout'] if args['dropout'] > 0. else 0.
        
        # parameters of the model
        
        self.container['params'], self.container['names'] = [], []
        
        self.container['embDict'] = OrderedDict()
        self.container['vars1'] = OrderedDict()
        self.container['dimIn1'] = 0
        
        self.container['vars2'] = OrderedDict()
        self.container['dimIn2'] = 0

        print '******************FEATURES1******************'
        for ed in self.args['features1']:
            if self.args['features1'][ed] == 0:
                self.container['embDict'][ed] = theano.shared( createMatrix(self.args['embs'][ed].astype(theano.config.floatX), self.args['kGivens'], ed) )
                
                if self.args['updateEmbs']:
                    print '@@@@@@@ Will update embedding tables'
                    self.container['params'] += [self.container['embDict'][ed]]
                    self.container['names'] += [ed]

            if self.args['features1'][ed] == 0:
                self.container['vars1'][ed] = T.imatrix()
                dimAdding = self.args['embs'][ed].shape[1]
                self.container['dimIn1'] += dimAdding        
            elif self.args['features1'][ed] == 1:
                self.container['vars1'][ed] = T.tensor3()
                dimAdding = self.args['features_dim1'][ed]
                self.container['dimIn1'] += dimAdding

            if self.args['features1'][ed] >= 0:
                print 'represetation1 - ', ed, ' : ', dimAdding 

        print 'REPRESENTATION1 DIMENSION = ', self.container['dimIn1']
        
        print '******************FEATURES2******************'
        for ed in self.args['features2']:
            if self.args['features2'][ed] == 0 and ed not in self.container['embDict']:
                self.container['embDict'][ed] = theano.shared( createMatrix(self.args['embs'][ed].astype(theano.config.floatX), self.args['kGivens'], ed) )
                
                if self.args['updateEmbs']:
                    print '@@@@@@@ Will update embedding tables'
                    self.container['params'] += [self.container['embDict'][ed]]
                    self.container['names'] += [ed]

            if self.args['features2'][ed] == 0:
                self.container['vars2'][ed] = T.imatrix()
                dimAdding = self.args['embs'][ed].shape[1]
                self.container['dimIn2'] += dimAdding        
            elif self.args['features2'][ed] == 1:
                self.container['vars2'][ed] = T.tensor3()
                dimAdding = self.args['features_dim2'][ed]
                self.container['dimIn2'] += dimAdding

            if self.args['features2'][ed] >= 0:
                print 'represetation2 - ', ed, ' : ', dimAdding 

        print 'REPRESENTATION2 DIMENSION = ', self.container['dimIn2']

        self.container['y'] = T.ivector('y') # label
        self.container['lr'] = T.scalar('lr')
        self.container['anchor1'] = T.ivector('anchorPosition1')
        self.container['anchor2'] = T.ivector('anchorPosition2')
        self.container['binaryFeatures1'] = T.imatrix('binaryFeatures1')
        self.container['binaryFeatures2'] = T.imatrix('binaryFeatures2')
        self.container['coreferenceFeatures'] = T.matrix('coreferenceFeatures')
        self.container['zeroVector'] = T.vector('zeroVector')
    
    def buildFunctions(self, p_y_given_x, p_y_given_x_dropout):
    
        if self.args['dropout'] == 0.:        
            nll = -T.mean(T.log(p_y_given_x)[T.arange(self.container['y'].shape[0]), self.container['y']])
        else:
            nll = -T.mean(T.log(p_y_given_x_dropout)[T.arange(self.container['y'].shape[0]), self.container['y']])
        
        if self.args['regularizer'] > 0.:
            for pp, nn in zip(self.container['params'], self.container['names']):
                if 'multi' in nn:
                    nll += self.args['regularizer'] * (pp ** 2).sum()
        
        y_pred = T.argmax(p_y_given_x, axis=1)
        
        gradients = T.grad( nll, self.container['params'] )

        classifyInput = [ self.container['vars1'][ed] for ed in self.args['features1'] if self.args['features1'][ed] >= 0 ] + [ self.container['vars2'][ed] for ed in self.args['features2'] if self.args['features2'][ed] >= 0 ]
        classifyInput += [ self.container['anchor1'], self.container['anchor2'] ]
        
        if self.args['coreferenceCutoff'] >= 0:
            classifyInput += [ self.container['coreferenceFeatures'] ]
        
        if self.args['binaryCutoff'] >= 0:
            classifyInput += [ self.container['binaryFeatures1'], self.container['binaryFeatures2'] ]
        
        # theano functions
        self.classify = theano.function(inputs=classifyInput, outputs=[y_pred, p_y_given_x], on_unused_input='ignore')

        trainInput = classifyInput + [self.container['y']]

        self.f_grad_shared, self.f_update_param = eval(self.args['optimizer'])(trainInput,nll,self.container['names'],self.container['params'],gradients,self.container['lr'],self.args['norm_lim'])
        
        self.container['setZero'] = OrderedDict()
        self.container['zeroVecs'] = OrderedDict()
        for ed in self.container['embDict']:
            self.container['zeroVecs'][ed] = numpy.zeros(self.args['embs'][ed].shape[1],dtype='float32')
            self.container['setZero'][ed] = theano.function([self.container['zeroVector']], updates=[(self.container['embDict'][ed], T.set_subtensor(self.container['embDict'][ed][0,:], self.container['zeroVector']))])

    def save(self, folder):
        storer = {}
        for param, name in zip(self.container['params'], self.container['names']):
            storer[name] = param.get_value()
        storer['binaryFeatureDict'] = self.args['binaryFeatureDict']
        storer['coreferenceFeatureDict'] = self.args['coreferenceFeatureDict']
        storer['window'] = self.args['window']
        sp = folder
        print 'saving parameters to: ', sp
        cPickle.dump(storer, open(sp, "wb"))
        #for param, name in zip(self.container['params'], self.container['names']):
        #    numpy.save(os.path.join(folder, name + '.npy'), param.get_value())

def localWordEmbeddingsTrigger(model, part):
    
    wedWindow = model.args['wedWindow']
    
    extendedWords = model.container['vars' + part]['word']
    wleft = T.zeros((extendedWords.shape[0], wedWindow), dtype='int32')
    wright = T.zeros((extendedWords.shape[0], wedWindow), dtype='int32')
    extendedWords = T.cast(T.concatenate([wleft, extendedWords, wright], axis=1), dtype='int32')
    
    def recurrence(words, pos, eembs):
        fet = words[pos:(pos+2*wedWindow+1)]
        fet = eembs[fet].flatten()
        return [fet]
    
    rep, _ = theano.scan(fn=recurrence, sequences=[extendedWords, model.container['anchor' + part]], n_steps=extendedWords.shape[0], non_sequences=[model.container['embDict']['word']], outputs_info=[None])
    
    dim_rep = (2*wedWindow+1) * model.args['embs']['word'].shape[1]
    
    return rep, dim_rep

class mainModel(BaseModel):
    def __init__(self, args):

        BaseModel.__init__(self, args)
        
        fetre1, dim_inter1 = eval(self.args['model'])(self, '1')
        
        if self.args['wedWindow'] > 0:
            rep1, dim_rep1 = localWordEmbeddingsTrigger(self, '1')
            fetre1 = T.concatenate([fetre1, rep1], axis=1)
            dim_inter1 += dim_rep1
        
        fetre_dropout1 = _dropout_from_layer(self.args['rng'], [fetre1], self.args['dropout'])
        fetre_dropout1 = fetre_dropout1[0]
            
        fetre2, dim_inter2 = eval(self.args['model'])(self, '2')
        
        if self.args['wedWindow'] > 0:
            rep2, dim_rep2 = localWordEmbeddingsTrigger(self, '2')
            fetre2 = T.concatenate([fetre2, rep2], axis=1)
            dim_inter2 += dim_rep2
        
        fetre_dropout2 = _dropout_from_layer(self.args['rng'], [fetre2], self.args['dropout'])
        fetre_dropout2 = fetre_dropout2[0]
        
        if self.args['coreferenceCutoff'] >= 0:
            fetre = T.concatenate([fetre1, fetre2, self.container['coreferenceFeatures']], axis=1)
            fetre_dropout = T.concatenate([fetre_dropout1, fetre_dropout2, self.container['coreferenceFeatures']], axis=1)
        
            dim_inter = dim_inter1 + dim_inter2 + self.args['coreferenceFeatureDim']
        else:
            fetre = T.concatenate([fetre1, fetre2], axis=1)
            fetre_dropout = T.concatenate([fetre_dropout1, fetre_dropout2], axis=1)
        
            dim_inter = dim_inter1 + dim_inter2
            
        hids = [dim_inter] + self.args['multilayerNN1']
        
        mul = MultiHiddenLayers([fetre, fetre_dropout], hids, self.container['params'], self.container['names'], 'multiMainModel', kGivens=self.args['kGivens'])
        
        fetre, fetre_dropout = mul[0], mul[1]
        
        dim_inter = hids[len(hids)-1]
        
        fW = theano.shared( createMatrix(randomMatrix(dim_inter, self.args['nc']), self.args['kGivens'], 'sofmaxMainModel_W') )
        fb = theano.shared( createMatrix(numpy.zeros(self.args['nc'], dtype=theano.config.floatX), self.args['kGivens'], 'sofmaxMainModel_b') )
        
        self.container['params'] += [fW, fb]
        self.container['names'] += ['sofmaxMainModel_W', 'sofmaxMainModel_b']
        
        p_y_given_x_dropout = T.nnet.softmax(T.dot(fetre_dropout, fW) + fb)
        
        p_y_given_x = T.nnet.softmax(T.dot(fetre , (1.0 - self.args['dropout']) * fW) + fb)
        
        self.buildFunctions(p_y_given_x, p_y_given_x_dropout)

class hybridModel(BaseModel):

    def __init__(self, args):

        BaseModel.__init__(self, args)
        
        fModel1, dim_model1 = eval(self.args['model'])(self, '1')
        
        if self.args['wedWindow'] > 0:
            rep1, dim_rep1 = localWordEmbeddingsTrigger(self, '1')
            fModel1 = T.concatenate([fModel1, rep1], axis=1)
            dim_model1 += dim_rep1
        
        fModel_dropout1 = _dropout_from_layer(self.args['rng'], [fModel1], self.args['dropout'])
        fModel_dropout1 = fModel_dropout1[0]
        
        fModel2, dim_model2 = eval(self.args['model'])(self, '2')
        
        if self.args['wedWindow'] > 0:
            rep2, dim_rep2 = localWordEmbeddingsTrigger(self, '2')
            fModel2 = T.concatenate([fModel2, rep2], axis=1)
            dim_model2 += dim_rep2
        
        fModel_dropout2 = _dropout_from_layer(self.args['rng'], [fModel2], self.args['dropout'])
        fModel_dropout2 = fModel_dropout2[0]
        
        if self.args['coreferenceCutoff'] >= 0:
            fModel = T.concatenate([fModel1, fModel2, self.container['coreferenceFeatures']], axis=1)
            fModel_dropout = T.concatenate([fModel_dropout1, fModel_dropout2, self.container['coreferenceFeatures']], axis=1)
        
            dim_model = dim_model1 + dim_model2 + self.args['coreferenceFeatureDim']
        else:
            fModel = T.concatenate([fModel1, fModel2], axis=1)
            fModel_dropout = T.concatenate([fModel_dropout1, fModel_dropout2], axis=1)
        
            dim_model = dim_model1 + dim_model2
        
        nnhids = [dim_model] + self.args['multilayerNN2']
        
        nnmul = MultiHiddenLayers([fModel, fModel_dropout], nnhids, self.container['params'], self.container['names'], 'multiHybridModelNN', kGivens=self.args['kGivens'])
        
        fModel, fModel_dropout = nnmul[0], nnmul[1]
        
        dim_model = nnhids[len(nnhids)-1]
        
        model_fW = theano.shared( createMatrix(randomMatrix(dim_model, self.args['nc']), self.args['kGivens'], 'multiHybridModelNN_sfW') )
        model_fb = theano.shared( createMatrix(numpy.zeros(self.args['nc'], dtype=theano.config.floatX), self.args['kGivens'], 'multiHybridModelNN_sfb') )
        
        self.container['params'] += [model_fW, model_fb]
        self.container['names'] += ['multiHybridModelNN_sfW', 'multiHybridModelNN_sfb']
        
        fModel_dropout = T.dot(fModel_dropout, model_fW) + model_fb
        fModel = T.dot(fModel , (1.0 - self.args['dropout']) * model_fW) + model_fb
        
        #-----multilayer nn
        
        hids = [self.args['binaryFeatureDim']] + self.args['multilayerNN1'] + [self.args['nc']]
        
        layer0_multi_W1 = theano.shared( createMatrix(randomMatrix(self.args['binaryFeatureDim'], hids[1]), self.args['kGivens'], 'l0_multiHybridModelBin_fW1') )
        layer0_multi_b1 = theano.shared( createMatrix(numpy.zeros(hids[1], dtype=theano.config.floatX), self.args['kGivens'], 'l0_multiHybridModelBin_fb1') )
        
        self.container['params'] += [layer0_multi_W1, layer0_multi_b1]
        self.container['names'] += ['l0_multiHybridModelBin_fW1', 'l0_multiHybridModelBin_fb1']
        
        def recurrence(bfi, Wmat, bvec):
            idx = bfi[1:(bfi[0]+1)]
            weights = T.sum(Wmat[idx], axis=0) + bvec
            return weights
        
        firstMapped1, _ = theano.scan(fn=recurrence, sequences=self.container['binaryFeatures1'], outputs_info=[None], non_sequences=[layer0_multi_W1, layer0_multi_b1], n_steps=self.container['binaryFeatures1'].shape[0])
        
        layer0_multi_W2 = theano.shared( createMatrix(randomMatrix(self.args['binaryFeatureDim'], hids[1]), self.args['kGivens'], 'l0_multiHybridModelBin_fW2') )
        layer0_multi_b2 = theano.shared( createMatrix(numpy.zeros(hids[1], dtype=theano.config.floatX), self.args['kGivens'], 'l0_multiHybridModelBin_fb2') )
        
        self.container['params'] += [layer0_multi_W2, layer0_multi_b2]
        self.container['names'] += ['l0_multiHybridModelBin_fW2', 'l0_multiHybridModelBin_fb2']
        
        firstMapped2, _ = theano.scan(fn=recurrence, sequences=self.container['binaryFeatures2'], outputs_info=[None], non_sequences=[layer0_multi_W2, layer0_multi_b2], n_steps=self.container['binaryFeatures2'].shape[0])
        
        firstMapped = firstMapped1 + firstMapped2
        
        if len(hids) == 2:
            fMulti = firstMapped
            fMulti_dropout = firstMapped
        else:
            firstMapped = T.nnet.sigmoid(firstMapped)
            hids = hids[1:(len(hids)-1)]
            fetreArr = MultiHiddenLayers([firstMapped], hids, self.container['params'], self.container['names'], 'multiHybridModelBin', kGivens=self.args['kGivens'])
            fMulti = fetreArr[0]
            dim_multi = hids[len(hids)-1]
            
            fW = theano.shared( createMatrix(randomMatrix(dim_multi, self.args['nc']), self.args['kGivens'], 'multiHybridModelBin_sfW') )
            fb = theano.shared( createMatrix(numpy.zeros(self.args['nc'], dtype=theano.config.floatX), self.args['kGivens'], 'multiHybridModelBin_sfb') )
        
            self.container['params'] += [fW, fb]
            self.container['names'] += ['multiHybridModelBin_sfW', 'multiHybridModelBin_sfb']
        
            fMulti = T.dot(fMulti , fW) + fb
            fMulti_dropout = fMulti
            
        fMulti = fMulti
        fMulti_dropout = fMulti
        
        fetre = fModel + fMulti
        fetre_dropout = fModel_dropout + fMulti_dropout
        
        su = T.exp(fetre - fetre.max(axis=1, keepdims=True))
        p_y_given_x = su / su.sum(axis=1, keepdims=True)
        
        su_dropout = T.exp(fetre_dropout - fetre_dropout.max(axis=1, keepdims=True))
        p_y_given_x_dropout = su_dropout / su_dropout.sum(axis=1, keepdims=True)
        
        self.buildFunctions(p_y_given_x, p_y_given_x_dropout)

def alternateHead(model, part):

    dimIn = model.container['dimIn' + part]
    _x = getConcatenation(model.container['embDict'], model.container['vars' + part], model.args['features' + part], model.args['features_dim' + part], tranpose=False)
    
    _x = convContextLs(_x, model.args['conv_feature_map'], model.args['conv_win_feature_map'], model.args['batch'], model.args['conv_winre'], dimIn, 'alternateHeadC' + part, model.container['params'], model.container['names'], kGivens=model.args['kGivens'])
    
    _x = _x.dimshuffle(1,0,2)
    _ix = _x[::-1]
    
    dimIn = model.args['conv_feature_map'] * len(model.args['conv_win_feature_map'])
    
    _x = gruBidirectCore(_x, _ix, dimIn, model.args['nh'], model.args['batch'], '_ab_alternateHeadR' + part, '_ab_ialternateHeadR' + part, model.container['params'], model.container['names'], kGivens=model.args['kGivens'])
    
    return rnnHeadIn(model, _x, 2, part)
    
def alternateHeadForward(model, part):

    dimIn = model.container['dimIn' + part]
    _x = getConcatenation(model.container['embDict'], model.container['vars' + part], model.args['features' + part], model.args['features_dim' + part], tranpose=False)
    
    _x = convContextLs(_x, model.args['conv_feature_map'], model.args['conv_win_feature_map'], model.args['batch'], model.args['conv_winre'], dimIn, 'alternateHeadForwardC' + part, model.container['params'], model.container['names'], kGivens=model.args['kGivens'])
    
    _x = _x.dimshuffle(1,0,2)
    
    dimIn = model.args['conv_feature_map'] * len(model.args['conv_win_feature_map'])
    
    _x = rnn_gru(_x, dimIn, model.args['nh'], model.args['batch'], '_ab_alternateHeadForwardR' + part, model.container['params'], model.container['names'], kGivens=model.args['kGivens'])
    
    _x = T.cast(_x.dimshuffle(1,0,2), dtype=theano.config.floatX)
    
    return rnnHeadIn(model, _x, 1, part)
    
def alternateHeadBackward(model, part):

    dimIn = model.container['dimIn' + part]
    _x = getConcatenation(model.container['embDict'], model.container['vars' + part], model.args['features' + part], model.args['features_dim' + part], tranpose=False)
    
    _x = convContextLs(_x, model.args['conv_feature_map'], model.args['conv_win_feature_map'], model.args['batch'], model.args['conv_winre'], dimIn, 'alternateHeadBackwardC' + part, model.container['params'], model.container['names'], kGivens=model.args['kGivens'])
    
    _x = _x.dimshuffle(1,0,2)[::-1]
    
    dimIn = model.args['conv_feature_map'] * len(model.args['conv_win_feature_map'])
    
    _x = rnn_gru(_x, dimIn, model.args['nh'], model.args['batch'], '_ab_alternateHeadBackwardR' + part, model.container['params'], model.container['names'], kGivens=model.args['kGivens'])
    
    _x = T.cast(_x[::-1].dimshuffle(1,0,2), dtype=theano.config.floatX)
    
    return rnnHeadIn(model, _x, 1, part)

def alternateMax(model, part):

    dimIn = model.container['dimIn' + part]
    _x = getConcatenation(model.container['embDict'], model.container['vars' + part], model.args['features' + part], model.args['features_dim' + part], tranpose=False)
    
    _x = convContextLs(_x, model.args['conv_feature_map'], model.args['conv_win_feature_map'], model.args['batch'], model.args['conv_winre'], dimIn, 'alternateMaxC' + part, model.container['params'], model.container['names'], kGivens=model.args['kGivens'])
    
    _x = _x.dimshuffle(1,0,2)
    _ix = _x[::-1]
    
    dimIn = model.args['conv_feature_map'] * len(model.args['conv_win_feature_map'])
    
    _x = gruBidirectCore(_x, _ix, dimIn, model.args['nh'], model.args['batch'], '_ab_alternateMaxR' + part, '_ab_ialternateMaxR' + part, model.container['params'], model.container['names'], kGivens=model.args['kGivens'])
    
    fRnn = T.cast(T.max(_x, axis=1), dtype=theano.config.floatX)
        
    dim_rnn = 2 * model.args['nh']
    
    return fRnn, dim_rnn

def alternateMaxForward(model, part):

    dimIn = model.container['dimIn' + part]
    _x = getConcatenation(model.container['embDict'], model.container['vars' + part], model.args['features' + part], model.args['features_dim' + part], tranpose=False)
    
    _x = convContextLs(_x, model.args['conv_feature_map'], model.args['conv_win_feature_map'], model.args['batch'], model.args['conv_winre'], dimIn, 'alternateMaxForwardC' + part, model.container['params'], model.container['names'], kGivens=model.args['kGivens'])
    
    _x = _x.dimshuffle(1,0,2)
    
    dimIn = model.args['conv_feature_map'] * len(model.args['conv_win_feature_map'])
    
    _x = rnn_gru(_x, dimIn, model.args['nh'], model.args['batch'], '_ab_alternateMaxForwardR' + part, model.container['params'], model.container['names'], kGivens=model.args['kGivens'])
    
    _x = T.cast(_x.dimshuffle(1,0,2), dtype=theano.config.floatX)
    
    fRnn = T.cast(T.max(_x, axis=1), dtype=theano.config.floatX)
        
    dim_rnn = model.args['nh']
    
    return fRnn, dim_rnn
    
def alternateMaxBackward(model, part):

    dimIn = model.container['dimIn' + part]
    _x = getConcatenation(model.container['embDict'], model.container['vars' + part], model.args['features' + part], model.args['features_dim' + part], tranpose=False)
    
    _x = convContextLs(_x, model.args['conv_feature_map'], model.args['conv_win_feature_map'], model.args['batch'], model.args['conv_winre'], dimIn, 'alternateMaxBackwardC' + part, model.container['params'], model.container['names'], kGivens=model.args['kGivens'])
    
    _x = _x.dimshuffle(1,0,2)[::-1]
    
    dimIn = model.args['conv_feature_map'] * len(model.args['conv_win_feature_map'])
    
    _x = rnn_gru(_x, dimIn, model.args['nh'], model.args['batch'], '_ab_alternateMaxBackwardR' + part, model.container['params'], model.container['names'], kGivens=model.args['kGivens'])
    
    _x = T.cast(_x[::-1].dimshuffle(1,0,2), dtype=theano.config.floatX)
    
    fRnn = T.cast(T.max(_x, axis=1), dtype=theano.config.floatX)
        
    dim_rnn = model.args['nh']
    
    return fRnn, dim_rnn
    
def alternateConv(model, part):

    _x = gruBiDirect(model.container['embDict'], model.container['vars' + part], model.args['features' + part], model.args['features_dim' + part], model.container['dimIn' + part], model.args['nh'], model.args['batch'], 'alternateConvR' + part, model.container['params'], model.container['names'], kGivens=model.args['kGivens'])
    
    dimIn = 2 * model.args['nh']
    
    fConv = convContext(_x, model.args['conv_feature_map'], model.args['conv_win_feature_map'], model.args['batch'], model.args['conv_winre'], dimIn, '_ab_alternateConvC' + part, model.container['params'], model.container['names'], kGivens=model.args['kGivens'])

    dim_conv = model.args['conv_feature_map'] * len(model.args['conv_win_feature_map'])
    
    return fConv, dim_conv
    
def alternateConvForward(model, part):

    _x = gruForward(model.container['embDict'], model.container['vars' + part], model.args['features' + part], model.args['features_dim' + part], model.container['dimIn' + part], model.args['nh'], model.args['batch'], 'alternateConvForwardR' + part, model.container['params'], model.container['names'], kGivens=model.args['kGivens'])
    
    dimIn = model.args['nh']
    
    fConv = convContext(_x, model.args['conv_feature_map'], model.args['conv_win_feature_map'], model.args['batch'], model.args['conv_winre'], dimIn, '_ab_alternateConvForwardC' + part, model.container['params'], model.container['names'], kGivens=model.args['kGivens'])

    dim_conv = model.args['conv_feature_map'] * len(model.args['conv_win_feature_map'])
    
    return fConv, dim_conv
    
def alternateConvBackward(model, part):

    _x = gruBackward(model.container['embDict'], model.container['vars' + part], model.args['features' + part], model.args['features_dim' + part], model.container['dimIn' + part], model.args['nh'], model.args['batch'], 'alternateConvBackwardR' + part, model.container['params'], model.container['names'], kGivens=model.args['kGivens'])
    
    dimIn = model.args['nh']
    
    fConv = convContext(_x, model.args['conv_feature_map'], model.args['conv_win_feature_map'], model.args['batch'], model.args['conv_winre'], dimIn, '_ab_alternateConvBackwardC' + part, model.container['params'], model.container['names'], kGivens=model.args['kGivens'])

    dim_conv = model.args['conv_feature_map'] * len(model.args['conv_win_feature_map'])
    
    return fConv, dim_conv
##
def convolute(model, part):
    _x = getConcatenation(model.container['embDict'], model.container['vars' + part], model.args['features' + part], model.args['features_dim' + part], tranpose=False)
        
    fConv = convContext(_x, model.args['conv_feature_map'], model.args['conv_win_feature_map'], model.args['batch'], model.args['conv_winre'], model.container['dimIn' + part], 'convolute' + part, model.container['params'], model.container['names'], kGivens=model.args['kGivens'])
        
    dim_conv = model.args['conv_feature_map'] * len(model.args['conv_win_feature_map'])
    
    return fConv, dim_conv
    
def nonConsecutiveConvolute(model, part):
    _x = getConcatenation(model.container['embDict'], model.container['vars' + part], model.args['features' + part], model.args['features_dim' + part], tranpose=False)
        
    fConv = nonConsecutiveConvNet(_x, model.args['conv_feature_map'], model.args['conv_win_feature_map'], model.args['batch'], model.args['conv_winre'], model.container['dimIn' + part], 'nonConsecutiveConvolute' + part, model.container['params'], model.container['names'], kGivens=model.args['kGivens'])
        
    dim_conv = model.args['conv_feature_map'] * len(model.args['conv_win_feature_map'])
    
    return fConv, dim_conv
    
def rnnHeadNonConsecutiveConv(model, part):
    rep_noncon, dim_noncon = nonConsecutiveConvolute(model, part)
    rep_rnn, dim_rnn = rnnHead(model, part)
    return T.concatenate([rep_noncon, rep_rnn], axis=1), dim_noncon + dim_rnn
##
def rnnHead(model, part):
    _x = gruBiDirect(model.container['embDict'], model.container['vars' + part], model.args['features' + part], model.args['features_dim' + part], model.container['dimIn' + part], model.args['nh'], model.args['batch'], 'rnnHead' + part, model.container['params'], model.container['names'], kGivens=model.args['kGivens'])
    return rnnHeadIn(model, _x, 2, part)
##  
def rnnHeadForward(model, part):
    _x = gruForward(model.container['embDict'], model.container['vars' + part], model.args['features' + part], model.args['features_dim' + part], model.container['dimIn' + part] , model.args['nh'], model.args['batch'], 'rnnHeadForward' + part, model.container['params'], model.container['names'], kGivens=model.args['kGivens'])
    return rnnHeadIn(model, _x, 1, part)
##
def rnnHeadBackward(model, part):
    _x = gruBackward(model.container['embDict'], model.container['vars' + part], model.args['features' + part], model.args['features_dim' + part], model.container['dimIn' + part], model.args['nh'], model.args['batch'], 'rnnHeadBackward' + part, model.container['params'], model.container['names'], kGivens=model.args['kGivens'])
    return rnnHeadIn(model, _x, 1, part)
##
def rnnHeadFf(model, part):
    _x = ffBiDirect(model.container['embDict'], model.container['vars' + part], model.args['features' + part], model.args['features_dim' + part], model.container['dimIn' + part] , model.args['nh'], model.args['batch'], 'rnnHeadFf' + part, model.container['params'], model.container['names'], kGivens=model.args['kGivens'])
    return rnnHeadIn(model, _x, 2, part)
##
def rnnHeadFfForward(model, part):
    _x = ffForward(model.container['embDict'], model.container['vars' + part], model.args['features' + part], model.args['features_dim' + part], model.container['dimIn' + part], model.args['nh'], model.args['batch'], 'rnnHeadFfForward' + part, model.container['params'], model.container['names'], kGivens=model.args['kGivens'])
    return rnnHeadIn(model, _x, 1, part)
##
def rnnHeadFfBackward(model, part):
    _x = ffBackward(model.container['embDict'], model.container['vars' + part], model.args['features' + part], model.args['features_dim' + part], model.container['dimIn' + part], model.args['nh'], model.args['batch'], 'rnnHeadFfBackward' + part, model.container['params'], model.container['names'], kGivens=model.args['kGivens'])
    return rnnHeadIn(model, _x, 1, part)
##
def rnnHeadIn(model, _x, num, part):
    
    def recurrence1(x_i, anchor):
        fet = x_i[anchor]
        return [fet]
    
    fRnn, _ = theano.scan(fn=recurrence1, sequences=[_x, model.container['anchor' + part]], outputs_info=[None], n_steps=_x.shape[0])
        
    dim_rnn = num * model.args['nh']
    
    return fRnn, dim_rnn
##
def rnnMax(model, part):
    _x = gruBiDirect(model.container['embDict'], model.container['vars' + part], model.args['features' + part], model.args['features_dim' + part], model.container['dimIn' + part], model.args['nh'], model.args['batch'], 'rnnMax' + part, model.container['params'], model.container['names'], kGivens=model.args['kGivens'])
    return rnnMaxIn(model, _x, 2)
##    
def rnnMaxForward(model, part):
    _x = gruForward(model.container['embDict'], model.container['vars' + part], model.args['features' + part], model.args['features_dim' + part], model.container['dimIn' + part], model.args['nh'], model.args['batch'], 'rnnMaxForward' + part, model.container['params'], model.container['names'], kGivens=model.args['kGivens'])
    return rnnMaxIn(model, _x, 1)
##
def rnnMaxBackward(model, part):
    _x = gruBackward(model.container['embDict'], model.container['vars' + part], model.args['features' + part], model.args['features_dim' + part], model.container['dimIn' + part], model.args['nh'], model.args['batch'], 'rnnMaxBackward' + part, model.container['params'], model.container['names'], kGivens=model.args['kGivens'])
    return rnnMaxIn(model, _x, 1)
##  
def rnnMaxFf(model, part):
    _x = ffBiDirect(model.container['embDict'], model.container['vars' + part], model.args['features' + part], model.args['features_dim' + part], model.container['dimIn' + part], model.args['nh'], model.args['batch'], 'rnnMaxFf' + part, model.container['params'], model.container['names'], kGivens=model.args['kGivens'])
    return rnnMaxIn(model, _x, 2)
##
def rnnMaxFfForward(model, part):
    _x = ffForward(model.container['embDict'], model.container['vars' + part], model.args['features' + part], model.args['features_dim' + part], model.container['dimIn' + part], model.args['nh'], model.args['batch'], 'rnnMaxFfForward' + part, model.container['params'], model.container['names'], kGivens=model.args['kGivens'])
    return rnnMaxIn(model, _x, 1)
##
def rnnMaxFfBackward(model, part):
    _x = ffBackward(model.container['embDict'], model.container['vars' + part], model.args['features' + part], model.args['features_dim' + part], model.container['dimIn' + part], model.args['nh'], model.args['batch'], 'rnnMaxFfBackward' + part, model.container['params'], model.container['names'], kGivens=model.args['kGivens'])
    return rnnMaxIn(model, _x, 1)

##
def rnnMaxIn(model, _x, num):
    fRnn = T.cast(T.max(_x, axis=1), dtype=theano.config.floatX)
        
    dim_rnn = num * model.args['nh']
    
    return fRnn, dim_rnn

######################
def rnnAtt(model, i):
    _x = gruBiDirect(model.container['embDict' + str(i)], model.container['vars' + str(i)], model.args['features' + str(i)], model.args['features_dim' + str(i)], model.container['dimIn' + str(i)] , model.args['nh' + str(i)], model.args['batch'], 'rnnAtt', model.container['params'], model.container['names'], outer=model.args['outer'])
    
    IW = theano.shared(randomMatrix(2 * model.args['nh' + str(i)], 1))
    Ib = theano.shared(numpy.zeros(1, dtype=theano.config.floatX))
        
    model.container['params'] += [IW, Ib]
    model.container['names'] += ['rnnAt_IW', 'rnnAT_Ib']     
        
    def recurrence(x_i):
        alpha = T.dot(x_i, IW) + Ib
        alpha = T.exp(alpha)
        alpha = alpha / T.sum(alpha)
        fet = (x_i * T.addbroadcast(alpha, 1).dimshuffle(0,'x')).sum(0)
        return [fet]
        
    fRnn, _ = theano.scan(fn=recurrence, \
            sequences=_x, outputs_info=[None], n_steps=_x.shape[0])
                
    dim_rnn = 2 * model.args['nh' + str(i)]
    
    return fRnn, dim_rnn
    
def rnnAttHead(model, i):
    _x = gruBiDirect(model.container['embDict' + str(i)], model.container['vars' + str(i)], model.args['features' + str(i)], model.args['features_dim' + str(i)], model.container['dimIn' + str(i)] , model.args['nh' + str(i)], model.args['batch'], 'rnnAtt', model.container['params'], model.container['names'], outer=model.args['outer'])
    
    IW = theano.shared(randomMatrix(2 * model.args['nh' + str(i)], 1))
    Ib = theano.shared(numpy.zeros(1, dtype=theano.config.floatX))
        
    model.container['params'] += [IW, Ib]
    model.container['names'] += ['rnnAt_IW', 'rnnAT_Ib']     
        
    def recurrenceAtt(x_i):
        alpha = T.dot(x_i, IW) + Ib
        alpha = T.exp(alpha)
        alpha = alpha / T.sum(alpha)
        fet = (x_i * T.addbroadcast(alpha, 1).dimshuffle(0,'x')).sum(0)
        return [fet]
        
    fRnnAtt, _ = theano.scan(fn=recurrenceAtt, \
            sequences=_x, outputs_info=[None], n_steps=_x.shape[0])
            
    def recurrenceHead(x_i, pos1, pos2):
        fet = T.cast(T.concatenate([x_i[pos1], x_i[pos2]]), dtype=theano.config.floatX)
        return [fet]
        
    fRnnHead, _ = theano.scan(fn=recurrenceHead, \
            sequences=[_x, model.container['pos1' + str(i)], model.container['pos2' + str(i)]], outputs_info=[None], n_steps=_x.shape[0])
    
    fRnn = T.cast(T.concatenate([fRnnAtt, fRnnHead], axis=1), dtype=theano.config.floatX)
                
    dim_rnn = 6 * model.args['nh' + str(i)]
    
    return fRnn, dim_rnn

#####################################
