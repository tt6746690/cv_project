import math
import numpy as np
import tensorflow as tf


############################################################
# dims
pixelnum = 608;
ms = [4];
frenums = [16];


##############################################################
# functions
def writeddata2d(data2d, name):
    n, m = data2d.shape
    fid = open(name, 'w')
    for i in range(n):
        for j in range(m):
            fid.write('%f ' % data2d[i, j])
        fid.write('\n')
    fid.close()
    return


def build_patterns_fft_old(patternsnm, frequency, n, m):
    patternsmn = tf.transpose(patternsnm);
    patternszero = tf.zeros_like(patternsmn)
    patternsmncomplex = tf.complex(patternsmn, patternszero)
    pfft = tf.fft(patternsmncomplex)
    frezero = tf.zeros_like(frequency)
    frecomplex = tf.complex(frequency, frezero)
    pfftnew = pfft * frecomplex;
    pcomplenew = tf.ifft(pfftnew);
    pmnnew = tf.real(pcomplenew)
    pnmnew = tf.transpose(pmnnew)
    return pnmnew


# frequency
def buildfftmtx(n, frenum):
    xnp = np.arange(n, dtype=np.float32);
    xnp = -xnp;
    xnp = 2.0 * math.pi * xnp / n;
    XXnp = np.tile(xnp.reshape(1, n), [n, 1])
    XXnp = XXnp * np.arange(n, dtype=np.float32).reshape(n, 1);
    XXcosnp = np.cos(XXnp);
    XXsinnp = np.sin(XXnp);
    #
    assert(frenum > 0)
    frearray = np.zeros(shape=(n,), dtype=np.bool);
    frearray[0] = True;
    frearray[1:frenum + 1] = True;
    frearray[-frenum:] = True;
    #
    KXcosnp = XXcosnp[frearray, :];
    KXsinnp = XXsinnp[frearray, :];
    XKcosnp = np.transpose(KXcosnp)
    XKsinnp = np.transpose(KXsinnp)
    return (np.matmul(XKcosnp, KXcosnp) + np.matmul(XKsinnp, KXsinnp)) / n


def build_patterns_fft(patternsnm, frenum, n, m):
    fremtxnp = buildfftmtx(n, frenum)
    fremtx = tf.constant(fremtxnp)
    return tf.matmul(fremtx, patternsnm)


eps = 1e-5
def build_as(cameras_bm):
    cam_hat = cameras_bm - tf.reduce_mean(cameras_bm, axis=1, keep_dims=True)
    return cam_hat + eps


def build_normal(cameras_bm):
    cam_len = tf.sqrt(tf.reduce_sum(tf.square(cameras_bm), axis=1, keep_dims=True))
    cam_norm = cameras_bm / cam_len
    return cam_norm


def build_asncc(cameras_bm):
    cam_hat = build_as(cameras_bm)
    cam_hat_norm = build_normal(cam_hat)
    return cam_hat_norm


def softmaxdep(score_bn, band_bn):
    scoresub = score_bn - tf.reduce_max(score_bn, axis=1, keep_dims=True);
    scoreexp = tf.exp(scoresub);
    scoretri = scoreexp * band_bn;
    scoresum = tf.reduce_sum(scoretri, axis=1, keep_dims=True);
    prob = scoretri / scoresum;
    return prob;


# blur
def gaussianblur(cam, kernel, m, n):
    camkm1b = tf.reshape(cam, shape=(-1, n, m));
    camkm1f = tf.nn.conv1d(camkm1b, kernel, stride=1, padding='SAME');
    return tf.reshape(camkm1f, shape=(-1, m));
