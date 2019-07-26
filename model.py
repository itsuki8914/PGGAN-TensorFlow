import tensorflow as tf
import numpy as np

def _EQfc_variable(weight_shape,name):
        with tf.variable_scope(name):
            # check weight_shape
            input_channels  = int(weight_shape[0])
            output_channels = int(weight_shape[1])
            c = np.sqrt(2.0 / input_channels)
            weight_shape    = (input_channels, output_channels)
            # define variables
            weight = tf.get_variable("w", weight_shape     ,
                                    initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0)) * c
            bias   = tf.get_variable("b", [weight_shape[1]],
                                    initializer=tf.constant_initializer(0.0))
        return weight, bias

def _EQconv_variable(weight_shape,name="conv"):
    with tf.variable_scope(name):
        # check weight_shape
        w = int(weight_shape[0])
        h = int(weight_shape[1])
        input_channels  = int(weight_shape[2])
        output_channels = int(weight_shape[3])
        c = np.sqrt(2.0 / (input_channels * w * h))
        weight_shape = (w,h,input_channels, output_channels)
        # define variables
        weight = tf.get_variable("w", weight_shape     ,
                                initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0)) * c
        bias   = tf.get_variable("b", [output_channels],
                                initializer=tf.constant_initializer(0.0))
    return weight, bias

def _conv2d(x, W, stride, padding="SAME"):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = padding)

def _EQconv(x, input_layer, output_layer, stride=1, filter_size=3, name="conv" ,padding ="SAME"):
    conv_w, conv_b = _EQconv_variable([filter_size,filter_size,input_layer,output_layer],name=name)
    h = _conv2d(x ,conv_w, stride=stride, padding=padding) + conv_b
    return h

def _EQfc(x, input_layer, output_layer, name="fc"):
    fc_w, fc_b = _EQfc_variable([input_layer,output_layer],name=name)
    h = tf.matmul(x, fc_w) + fc_b
    return h

def pixel_norm(x, epsilon=1e-8):
    return x / tf.sqrt(tf.reduce_mean(tf.square(x), axis=3, keepdims=True) + epsilon)

def _avg_pool(x, k=2):
    return tf.nn.avg_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

def minibatch_std(x):
    x_b, x_h, x_w, x_c = x.get_shape().as_list()
    group_size = tf.minimum(4, tf.shape(x)[0])
    y = tf.reshape(x, [group_size, -1, x_h, x_w, x_c]) #[GMHWC]
    mean = tf.reduce_mean(y, axis=0, keepdims=True)    #[GMHWC]
    y = y - mean                                       #[GMHWC]
    var = tf.reduce_mean(tf.square(y), axis=0)         #[MHWC]
    std = tf.sqrt(var + 1e-8)                          #[MHWC]
    ave = tf.reduce_mean(std, axis=[1,2,3], keepdims=True) # [M111]
    fmap = tf.tile(ave, [group_size, x_h, x_w, 1]) # [NHW1]
    return tf.concat([x, fmap], axis=3)

def _up_sampling(x, ratio=2):
    x_b, x_h, x_w, x_c = x.get_shape().as_list()
    return tf.image.resize_bilinear(x, [x_h*ratio, x_w*ratio])

def flatten(x):
    n_b, n_h, n_w, n_f = [int(x) for x in h.get_shape()]
    h = tf.reshape(h,[-1,n_h*n_w*n_f])
    return h

def buildGenerator(z, alpha, stage):
    fn = [512, 512, 512, 512, 512, 256, 128, 64, 32, 16]
    with tf.variable_scope("Generator") as scope:
        for i in range(1, stage+1):
            reuse = False if stage == i else True
            if i==1:
                with tf.variable_scope("stage1", reuse=reuse):
                    h = tf.pad(z, [[0, 0], [3, 3], [3, 3], [0, 0]], 'CONSTANT')
                    h = _EQconv(h, fn[i-1], fn[i], 1, 4, "conv1" , padding ="VALID")
                    h = pixel_norm(h)
                    h = tf.nn.leaky_relu(h)

                    h = _EQconv(h, fn[i], fn[i], 1, 3, "conv2")
                    h = pixel_norm(h)
                    h = tf.nn.leaky_relu(h)

                    rgb = _EQconv(h, fn[i], 3, 1, 1, "toRGB")
            else:
                with tf.variable_scope("stage%d"%i, reuse=reuse):
                    h = _up_sampling(h)

                    shortcut =  _EQconv(h, fn[i-1], 3, 1, 1, "shortcut")

                    h = _EQconv(h, fn[i-1], fn[i], 1, 3, "conv1")
                    h = pixel_norm(h)
                    h = tf.nn.leaky_relu(h)

                    h = _EQconv(h, fn[i], fn[i], 1, 3, "conv2")
                    h = pixel_norm(h)
                    h = tf.nn.leaky_relu(h)

                    rgb = _EQconv(h, fn[i], 3, 1, 1, "toRGB")

                    if stage == i:
                        rgb = rgb * alpha + shortcut * (1 - alpha)
    y = tf.nn.tanh(rgb)
    #print(y)
    return y

def buildDiscriminator(y, alpha, stage, reuse):
    fn = [512, 512, 512, 512, 512, 256, 128, 64, 32, 16]
    with tf.variable_scope("Discriminator") as scope:
        for i in range(stage, 1, -1):
            _reuse = reuse if stage == i else True
            with tf.variable_scope("stage%d"%i, reuse=_reuse):
                if i == stage:
                    shortcut = _avg_pool(y)
                    shortcut = _EQconv(shortcut, 3, fn[i-1], 1, 3, "shortcut")
                    shortcut = tf.nn.leaky_relu(shortcut)

                    h = _EQconv(y, 3, fn[i], 1, 3, "fromRGB")
                    h = tf.nn.leaky_relu(h)

                    h = _EQconv(h, fn[i], fn[i], 1, 3, "conv1")
                    h = tf.nn.leaky_relu(h)

                    h = _EQconv(h, fn[i], fn[i-1], 1, 3, "conv2")
                    h = tf.nn.leaky_relu(h)

                    h = _avg_pool(h)
                    h = h * alpha + shortcut * (1 - alpha)
                else:
                    h = _EQconv(h, fn[i], fn[i], 1, 3, "conv1")
                    h = tf.nn.leaky_relu(h)

                    h = _EQconv(h, fn[i], fn[i-1], 1, 3, "conv2")
                    h = tf.nn.leaky_relu(h)

                    h = _avg_pool(h)

        _reuse = reuse if stage == 1 else True
        with tf.variable_scope("stage1", reuse=_reuse):
            if stage == 1:
                h = _EQconv(y, 3, fn[1], 1, 3, "fromRGB")
                h = tf.nn.leaky_relu(h)
            h = minibatch_std(h)

            h = _EQconv(h, fn[1]+1, fn[1], 1, 3, "conv1")
            h = tf.nn.leaky_relu(h)

            h = _EQconv(h, fn[1], fn[0], 1, 3, "conv2")
            h = tf.nn.leaky_relu(h)

            n_b, n_h, n_w, n_f = h.get_shape().as_list()
            h = tf.reshape(h,[-1,n_h*n_w*n_f])
            h = _EQfc(h, n_h*n_w*n_f, 1, "fc1")

    return h
