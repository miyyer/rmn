import theano, cPickle, h5py, lasagne, random, csv, gzip                                                  
import numpy as np
import theano.tensor as T         


# rewritten embedding layer
class MyEmbeddingLayer(lasagne.layers.Layer):
    
    def __init__(self, incoming, input_size, output_size,
                 W=lasagne.init.Normal(), name='W', **kwargs):
        super(MyEmbeddingLayer, self).__init__(incoming, **kwargs)

        self.input_size = input_size
        self.output_size = output_size

        self.W = self.add_param(W, (input_size, output_size), name=name)

    def get_output_shape_for(self, input_shape):
        return input_shape + (self.output_size, )

    def get_output_for(self, input, **kwargs):
        return self.W[input]

# compute vector average
class AverageLayer(lasagne.layers.MergeLayer):
    def __init__(self, incomings, d, **kwargs):
        super(AverageLayer, self).__init__(incomings, **kwargs)
        self.d = d
        self.sum = True

    def get_output_for(self, inputs, **kwargs):
        emb_sums = T.sum(inputs[0] * inputs[1][:, :, None], axis=1)
        if self.sum:
            return emb_sums
        else:
            mask_sums = T.sum(inputs[1], axis=1)
            return emb_sums / mask_sums[:,None]

    # batch_size x max_spans x d
    def get_output_shape_for(self, input_shapes):
        return (None, self.d)


# multiply recurrent hidden states with descriptor matrix R
class ReconLayer(lasagne.layers.Layer):
    def __init__(self, incoming, d, num_descs, **kwargs):
        super(ReconLayer, self).__init__(incoming, **kwargs)
        self.R = self.add_param(lasagne.init.GlorotUniform(), 
            (num_descs, d), name='R')
        self.d = d

    def get_output_for(self, hiddens, **kwargs):
        return T.dot(hiddens, self.R)
        
    # batch_size x max_spans x d
    def get_output_shape_for(self, input_shapes):
        return (None, self.d)


# mix word embeddings with global character/book embeddings
class MixingLayer(lasagne.layers.MergeLayer):
    def __init__(self, incomings, d, d_char, 
        d_book, **kwargs):

        super(MixingLayer, self).__init__(incomings, **kwargs)
        self.d = d
        self.W_m = self.add_param(lasagne.init.GlorotUniform(), 
            (d, d), name='W_m')
        self.b_m = self.add_param(lasagne.init.Constant(0), 
            (d,), name='b_m')
        self.W_char = self.add_param(lasagne.init.GlorotUniform(), 
            (d, d_char), name='W_char') 
        self.W_book = self.add_param(lasagne.init.GlorotUniform(), 
            (d, d_book), name='W_book') 
        self.f = lasagne.nonlinearities.rectify

    def get_output_for(self, inputs, **kwargs):
        spanvec = T.dot(inputs[0], self.W_m) + self.b_m
        cvec = T.dot(self.W_char, T.sum(inputs[1], axis=0))
        bvec = T.dot(self.W_book, T.flatten(inputs[2]))
        return self.f(spanvec + cvec[None, :] + bvec[None, :])

    # num_spans, self.d
    def get_output_shape_for(self, input_shapes):
        return (None, self.d)


# recurrent layer with linear interpolation
class RecurrentRelationshipLayer(lasagne.layers.Layer):
    def __init__(self, incoming, d_word, 
        d_hidden, num_descs, **kwargs):

        super(RecurrentRelationshipLayer, self).__init__(incoming, **kwargs)
        self.W = self.add_param(lasagne.init.GlorotUniform(), 
            (d_word, num_descs), name='W_arch')
        self.W2 = self.add_param(lasagne.init.GlorotUniform(), 
            (num_descs, num_descs), name='W2_arch')

        self.d_word = d_word
        self.d_hidden = d_hidden
        self.num_descs = num_descs
        self.alpha = 0.5
        self.f = lasagne.nonlinearities.softmax

    def get_output_for(self, proj, **kwargs):

        # recurrence over spans in a single relationship
        def step_fn(curr, r, h_prev):    
            
            if T.gt(r, 0):
                curr_hid = self.alpha * h_prev
                curr_hid += (1 - self.alpha) *\
                    self.f(curr + T.dot(self.W2, h_prev))

            else:
                curr_hid = self.f(curr)

            curr_hid = T.flatten(curr_hid)
            return curr_hid

        # precompute input
        precomp = T.dot(proj, self.W)
        r = T.arange(precomp.shape[0])

        hiddens, updates = theano.scan(
            step_fn, 
            sequences=[precomp, r], 
            outputs_info=T.zeros(self.num_descs), ) 
        
        return hiddens

    # batch_size x d
    def get_output_shape_for(self, input_shape):
        return (None, self.num_descs)