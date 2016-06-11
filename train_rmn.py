import theano, cPickle, h5py, lasagne, random, csv, gzip, time                                                  
import numpy as np
import theano.tensor as T 
from layers import *
from util import *        

# assemble the network
def build_rmn(d_word, d_char, d_book, d_hidden, len_voc, 
    num_descs, num_chars, num_books, span_size, We, 
    freeze_words=True, eps=1e-5, lr=0.01, negs=10):

    # input theano vars
    in_spans = T.imatrix(name='spans')
    in_neg = T.imatrix(name='neg_spans')
    in_chars = T.ivector(name='chars')
    in_book = T.ivector(name='books')
    in_currmasks = T.matrix(name='curr_masks')
    in_dropmasks = T.matrix(name='drop_masks')
    in_negmasks = T.matrix(name='neg_masks')

    # define network
    l_inspans = lasagne.layers.InputLayer(shape=(None, span_size), 
        input_var=in_spans)
    l_inneg = lasagne.layers.InputLayer(shape=(negs, span_size), 
        input_var=in_neg)
    l_inchars = lasagne.layers.InputLayer(shape=(2, ), 
        input_var=in_chars)
    l_inbook = lasagne.layers.InputLayer(shape=(1, ), 
        input_var=in_book)
    l_currmask = lasagne.layers.InputLayer(shape=(None, span_size), 
        input_var=in_currmasks)
    l_dropmask = lasagne.layers.InputLayer(shape=(None, span_size), 
        input_var=in_dropmasks)
    l_negmask = lasagne.layers.InputLayer(shape=(negs, span_size), 
        input_var=in_negmasks)

    # negative examples should use same embedding matrix
    l_emb = MyEmbeddingLayer(l_inspans, len_voc, 
        d_word, W=We, name='word_emb')
    l_negemb = MyEmbeddingLayer(l_inneg, len_voc, 
            d_word, W=l_emb.W, name='word_emb_copy1')

    # freeze embeddings
    if freeze_words:
        l_emb.params[l_emb.W].remove('trainable')
        l_negemb.params[l_negemb.W].remove('trainable')

    l_chars = lasagne.layers.EmbeddingLayer(\
        l_inchars, num_chars, d_char, name='char_emb')
    l_books = lasagne.layers.EmbeddingLayer(\
        l_inbook, num_books, d_book, name='book_emb')

    # average each span's embeddings
    l_currsum = AverageLayer([l_emb, l_currmask], d_word)
    l_dropsum = AverageLayer([l_emb, l_dropmask], d_word)
    l_negsum = AverageLayer([l_negemb, l_negmask], d_word)

    # pass all embeddings thru feed-forward layer
    l_mix = MixingLayer([l_dropsum, l_chars, l_books],
        d_word, d_char, d_book)

    # compute recurrent weights over dictionary
    l_rels = RecurrentRelationshipLayer(\
        l_mix, d_word, d_hidden, num_descs)

    # multiply weights with dictionary matrix
    l_recon = ReconLayer(l_rels, d_word, num_descs)

    # compute loss
    currsums = lasagne.layers.get_output(l_currsum)
    negsums = lasagne.layers.get_output(l_negsum)
    recon = lasagne.layers.get_output(l_recon)

    currsums /= currsums.norm(2, axis=1)[:, None]
    recon /= recon.norm(2, axis=1)[:, None]
    negsums /= negsums.norm(2, axis=1)[:, None]
    correct = T.sum(recon * currsums, axis=1)
    negs = T.dot(recon, negsums.T)
    loss = T.sum(T.maximum(0., 
        T.sum(1. - correct[:, None] + negs, axis=1)))

    # enforce orthogonality constraint
    norm_R = l_recon.R / l_recon.R.norm(2, axis=1)[:, None]
    ortho_penalty = eps * T.sum((T.dot(norm_R, norm_R.T) - \
        T.eye(norm_R.shape[0])) ** 2)
    loss += ortho_penalty

    all_params = lasagne.layers.get_all_params(l_recon, trainable=True)
    updates = lasagne.updates.adam(loss, all_params, learning_rate=lr)
    traj_fn = theano.function([in_chars, in_book, 
        in_spans, in_dropmasks], 
        lasagne.layers.get_output(l_rels))
    train_fn = theano.function([in_chars, in_book, 
        in_spans, in_currmasks, in_dropmasks,
        in_neg, in_negmasks], 
        [loss, ortho_penalty], updates=updates)
    return train_fn, traj_fn, l_recon

if __name__ == '__main__':

    print 'loading data...'
    span_data, span_size, wmap, cmap, bmap = \
        load_data('data/relationships.csv.gz', 'data/metadata.pkl')
    We = cPickle.load(open('data/glove.We', 'rb')).astype('float32')
    norm_We = We / np.linalg.norm(We, axis=1)[:, None]
    We = np.nan_to_num(norm_We)
    descriptor_log = 'models/descriptors.log'
    trajectory_log = 'models/trajectories.log'

    # embedding/hidden dimensionality
    d_word = We.shape[1]
    d_char = 50
    d_book = 50
    d_hidden = 50

    # number of descriptors
    num_descs = 30

    # number of negative samples per relationship
    num_negs = 50

    # word dropout probability
    p_drop = 0.75
    
    n_epochs = 15
    lr = 0.001
    eps = 1e-6
    num_chars = len(cmap)
    num_books = len(bmap)
    num_traj = len(span_data)
    len_voc = len(wmap)
    revmap = {}
    for w in wmap:
        revmap[wmap[w]] = w

    print d_word, span_size, num_descs, len_voc,\
        num_chars, num_books, num_traj

    print 'compiling...'
    train_fn, traj_fn, final_layer = build_rmn(
        d_word, d_char, d_book, d_hidden, len_voc, num_descs, num_chars, 
        num_books, span_size, We, eps=eps, 
        freeze_words=True, lr=lr, negs=num_negs)
    print 'done compiling, now training...'

    # training loop
    min_cost = float('inf')
    for epoch in range(n_epochs):
        cost = 0.
        random.shuffle(span_data)
        start_time = time.time()
        for book, chars, curr, cm, in span_data:
            ns, nm = generate_negative_samples(\
                num_traj, span_size, num_negs, span_data)

            # word dropout
            drop_mask = (np.random.rand(*(cm.shape)) < (1 - p_drop)).astype('float32')
            drop_mask *= cm

            ex_cost, ex_ortho = train_fn(chars, book, curr, cm, drop_mask,
                ns, nm)
            cost += ex_cost
        end_time = time.time()

        # save params if cost went down
        if cost < min_cost:
            min_cost = cost
            params = lasagne.layers.get_all_params(final_layer)
            p_values = [p.get_value() for p in params]
            p_dict = dict(zip([str(p) for p in params], p_values))
            cPickle.dump(p_dict, open('models/rmn_params.pkl', 'wb'),
                protocol=cPickle.HIGHEST_PROTOCOL)

            # compute nearest neighbors of descriptors
            R = p_dict['R']
            log = open(descriptor_log, 'w')
            for ind in range(len(R)):
                desc = R[ind] / np.linalg.norm(R[ind])
                sims = We.dot(desc.T)
                ordered_words = np.argsort(sims)[::-1]
                desc_list = [ revmap[w] for w in ordered_words[:10]]
                log.write(' '.join(desc_list) + '\n')
                print 'descriptor %d:' % ind
                print desc_list
            log.flush()
            log.close()

            # save relationship trajectories
            print 'writing trajectories...'
            tlog = open(trajectory_log, 'wb')
            traj_writer = csv.writer(tlog)
            traj_writer.writerow(['Book', 'Char 1', 'Char 2', 'Span ID'] + \
                ['Topic ' + str(i) for i in range(num_descs)])
            for book, chars, curr, cm in span_data:
                c1, c2 = [cmap[c] for c in chars]
                bname = bmap[book]

                # feed unmasked inputs to get trajectories
                traj = traj_fn(chars, book, curr, cm)
                for ind in range(len(traj)):
                    step = traj[ind]
                    traj_writer.writerow([bname, c1, c2, ind] + \
                    list(step) )   

            tlog.flush()
            tlog.close()

        print 'done with epoch: ', epoch, ' cost =',\
            cost / len(span_data), 'time: ', end_time-start_time