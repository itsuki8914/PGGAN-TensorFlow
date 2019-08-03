import os,sys,shutil
import tensorflow as tf
import numpy as np
import argparse
import cv2,math,glob,random,time
import time
import matplotlib.pyplot as plt
from model import *
from btgen import BatchGenerator

DATASET_DIR = "data"
SAVE_DIR = "model"
SVIM_DIR = "sample"

def tileImage(imgs):
    d = int(math.sqrt(imgs.shape[0]-1))+1
    h = imgs[0].shape[0]
    w = imgs[0].shape[1]
    r = np.zeros((h*d,w*d,3),dtype=np.float32)
    for idx,img in enumerate(imgs):
        idx_y = int(idx/d)
        idx_x = idx-idx_y*d
        r[idx_y*h:(idx_y+1)*h,idx_x*w:(idx_x+1)*w,:] = img
    return r

def printParam(scope):
    total_parameters = 0
    for variable in tf.trainable_variables(scope=scope):
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print("{} has {} parameters".format(scope, total_parameters))

def calc_losses(d_reals, d_fakes, xhats, d_xhats):
    g_losses = []
    d_losses = []
    for d_real, d_fake, xhat, d_xhat in zip(d_reals, d_fakes, xhats, d_xhats):
        g_loss = -tf.reduce_mean(d_fake)
        d_loss = tf.reduce_mean(d_fake) - tf.reduce_mean(d_real)

        drift_loss = tf.reduce_mean(d_real ** 2 * 1e-3)
        d_loss += drift_loss

        scale = 10.0
        grad = tf.gradients(d_xhat, [xhat])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1, 2, 3]))
        gradient_penalty = tf.reduce_mean(tf.square(slopes - 1.0) * scale)
        d_loss += gradient_penalty

        g_losses.append(g_loss)
        d_losses.append(d_loss)

    return g_losses, d_losses

def main():
    if not os.path.exists(SAVE_DIR):
        os.mkdir(SAVE_DIR)

    if not os.path.exists(SVIM_DIR):
        os.mkdir(SVIM_DIR)

    img_size = [2**(i+2) for i in range(9)]
    bs = [64, 64, 32, 32, 32, 16, 8, 4, 4]
    steps = [8000,10000,20000,40000,50000,60000,80000,90000,100000]
    z_dim = 512
    lmd = 10

    batch = BatchGenerator(img_size=256,datadir=DATASET_DIR)
    IN_ = batch.getBatch(4)
    IN_ = (IN_ + 1)*127.5
    IN_ =tileImage(IN_)
    cv2.imwrite("{}/input.png".format(SVIM_DIR),IN_)

    z = tf.placeholder(tf.float32, [None, 1, 1, z_dim])
    X_real =  [tf.placeholder(tf.float32, [None, r, r, 3]) for r in img_size]
    alpha = tf.placeholder(tf.float32, [])
    X_fake = [buildGenerator(z, alpha, stage=i+1) for i in range(9)]
    fake_y = [buildDiscriminator(x, alpha, stage=i+1, reuse=False) for i, x in enumerate(X_fake)]
    real_y = [buildDiscriminator(x, alpha, stage=i+1, reuse=True) for i, x in enumerate(X_real)]

    #WGAN-GP
    xhats = []
    d_xhats = []
    for i, (real, fake) in enumerate(zip(X_real, X_fake)):
        epsilon = tf.random_uniform(shape=[tf.shape(real)[0], 1, 1, 1], minval=0.0, maxval=1.0)
        inter = real * epsilon + fake * (1 - epsilon)
        d_xhat = buildDiscriminator(inter, alpha, stage=i+1, reuse=True)
        xhats.append(inter)
        d_xhats.append(d_xhat)

    g_losses, d_losses = calc_losses(real_y, fake_y, xhats, d_xhats)

    g_var = [x for x in tf.trainable_variables() if "Generator"     in x.name]
    d_var = [x for x in tf.trainable_variables() if "Discriminator" in x.name]
    opt = tf.train.AdamOptimizer(learning_rate=1e-3, beta1=0.0, beta2=0.99, epsilon=1e-8)

    g_opt = [opt.minimize(g_loss, var_list=g_var) for g_loss in g_losses]
    d_opt = [opt.minimize(d_loss, var_list=d_var) for d_loss in d_losses]

    printParam(scope="Generator")
    printParam(scope="Discriminator")

    start = time.time()

    config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.75))

    sess =tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()

    ckpt = tf.train.get_checkpoint_state(SAVE_DIR)

    if ckpt: # checkpointがある場合
        last_model = ckpt.model_checkpoint_path # 最後に保存したmodelへのパス
        print ("load " + last_model)
        saver.restore(sess, last_model) # 変数データの読み込み
        print("succeed restore model")
    else:
        print("models were not found")
        init = tf.global_variables_initializer()
        sess.run(init)

    print("%.4e sec took initializing"%(time.time()-start))

    start = time.time()
    for stage in range(0,9):
        batch =  BatchGenerator(img_size=img_size[stage],datadir=DATASET_DIR)

        x_batch = batch.getBatch(bs[stage],alpha=1.0)
        out = tileImage(x_batch)
        out = np.array((out + 1) * 127.5, dtype=np.uint8)
        outdir = os.path.join(SVIM_DIR, 'stage{}'.format(stage+1))
        os.makedirs(outdir, exist_ok=True)
        dst = os.path.join(outdir, 'sample.png')
        cv2.imwrite(dst, out)
        g_hist = []
        d_hist = []
        print("starting stage{}".format(stage+1))
        for i in range(steps[stage]+1):
            delta = 4*i/(steps[stage])
            if stage == 1:
                alp = 1.0
            else:
                alp = min(delta, 1.0)

            x_batch = batch.getBatch(bs[stage],alpha=alp)

            z_batch = np.random.normal(0, 0.5, [bs[stage], 1, 1, 512])

            _, dis_loss = sess.run([d_opt[stage], d_losses[stage]],
                                 feed_dict={X_real[stage]: x_batch, z: z_batch, alpha: alp})

            z_batch = np.random.normal(0, 0.5, [bs[stage], 1, 1, 512])
            _, gen_loss = sess.run([g_opt[stage], g_losses[stage]], feed_dict={z: z_batch, alpha: alp})

            g_hist.append(gen_loss)
            d_hist.append(dis_loss)

            print("in step %s, dis_loss = %.4e, gen_loss = %.4e" %(i,dis_loss, gen_loss))

            if i%100 == 0:
                # save sample image
                z_batch = np.random.normal(0, 0.5, [bs[stage], 1, 1, 512])
                out = X_fake[stage].eval(feed_dict={z: z_batch, alpha: alp}, session=sess)
                out = tileImage(out)
                out = np.array((out + 1) * 127.5, dtype=np.uint8)
                outdir = os.path.join(SVIM_DIR, 'stage{}'.format(stage+1))
                os.makedirs(outdir, exist_ok=True)
                dst = os.path.join(outdir, '{}.png'.format('{0:09d}'.format(i)))
                cv2.imwrite(dst, out)

                # save loss graph
                fig = plt.figure(figsize=(8,6), dpi=128)
                ax = fig.add_subplot(111)
                plt.title("Loss")
                plt.grid(which="both")
                ax.plot(g_hist,label="gen_loss", linewidth = 0.25)
                ax.plot(d_hist,label="dis_loss", linewidth = 0.25)
                plt.xlabel('step', fontsize = 16)
                plt.ylabel('loss', fontsize = 16)
                plt.legend(loc = 'upper right')
                plt.savefig(os.path.join(outdir,"hist.png"))
                plt.close()

            if i % 5000 == 0 and i!=0:
                saver.save(sess,os.path.join(SAVE_DIR,"model.ckpt"),i)

if __name__ == '__main__':
    main()
