import sys
sys.path.append('../')

import numpy as np
import tensorflow as tf
from utils import *
import os


###############################################
# dims
'''
m = 5
frenum = 8
'''
n = pixelnum
bt = 1;
batch = n * bt


##############################################
# gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def optim(m, frenum, t):
    # variables
    patterns = tf.Variable(tf.random_uniform([n, m], 0, 1, dtype=tf.float32))
    # placeholder
    albedo = tf.placeholder(tf.float32, shape=[batch, 1])
    B = tf.placeholder(tf.float32, shape=[batch, 1])
    noise = tf.placeholder(tf.float32, shape=[batch, m])
    noise2 = tf.placeholder(tf.float32, shape=[batch, m])
    pat_idx = tf.placeholder(tf.int32, shape=[batch, ])
    # frequency = tf.placeholder(tf.float32, shape=[m, n])
    gtmap = tf.placeholder(tf.float32, shape=[batch, n])
    # process
    pft01 = tf.clip_by_value(patterns, 0, 1);
    pft = build_patterns_fft(pft01, frenum, n, m)
    patclip = tf.clip_by_value(pft, clip_value_min=0, clip_value_max=1)
    pat_sele = tf.nn.embedding_lookup(patclip, pat_idx)
    camraw = albedo * pat_sele + B;
    # cam = camraw + tf.sqrt(camraw) * noise2 + noise
    cam = camraw + camraw * noise2 + noise
    camclip = tf.clip_by_value(cam, 0, 1)
    camasncc = build_asncc(camclip)
    patasncc = build_asncc(patclip)
    # probability
    s = tf.matmul(camasncc, patasncc, transpose_b=True)
    prob = tf.nn.softmax(200 * s)
    loss = 1 - tf.reduce_sum(gtmap * prob) / batch
    # loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cam2, labels=pat_idx));
    # optimizer
    learning_rate = tf.placeholder(tf.float32, shape=[])
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, beta2=0.999).minimize(loss, var_list=(patterns))
    # sess = tf.Session()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    # run
    iternum = 1500
    for i in range(iternum):
        if i < 1000:
            lr = 0.01
        else:
            lr = 0.003
        # input
        albedonp = 1.0 * np.random.rand(batch, 1).astype(np.float32)
        Bnp = 0.1 * np.random.rand(batch, 1).astype(np.float32)
        noisenp = 0.003 * np.random.randn(batch, m).astype(np.float32)
        noise2np = 0.02 * np.random.randn(batch, m).astype(np.float32)
        idxnp = np.tile(np.random.permutation(n), bt);
        gtmapnp = np.zeros((batch, n));
        gtmapnp[np.arange(batch), idxnp] = 1;
        _, loss1_, patternsnp, poss = sess.run([optimizer, loss, patterns, s],
                                               feed_dict={albedo: albedonp, B: Bnp, noise: noisenp, noise2: noise2np,
                                                          pat_idx: idxnp, gtmap:gtmapnp, learning_rate: lr})
        # ratio = np.sum(np.arsort(poss)[:, -1] == idxnp) * 1.0 / batch
        ratio = np.sum(np.argmax(poss, 1) == idxnp) * 1.0 / batch
        print('%d_%d %d %f %f %f\n' % (m, frenum, i, loss1_, ratio, patternsnp[0, 0]))
    patternsnp, pclipnp = sess.run([patterns, patclip]);
    savefolder = 'asncc1_%d_%d' % (t, n)
    if os.path.isdir(savefolder):
        pass
    else:
        os.mkdir(savefolder)
    writeddata2d(patternsnp, '%s/patterns-%d-%d-%d.txt' % (savefolder, n, m, frenum))
    writeddata2d(pclipnp, '%s/pt01-%d-%d-%d.txt' % (savefolder, n, m, frenum))


import time
for t in range(5):
    for m_ in ms:
        for frenum_ in frenums:
            # print m_, frenum_
            localtime1 = time.localtime(time.time())
            optim(m_, frenum_, t)
            localtime2 = time.localtime(time.time())
            print "time1:", localtime1
            print "time2:", localtime2
