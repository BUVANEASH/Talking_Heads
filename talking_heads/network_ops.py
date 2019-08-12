# -*- coding: utf-8 -*-
#/usr/bin/python3

import tensorflow as tf

##################################################################################
# Initialization
##################################################################################

# Xavier : tf_contrib.layers.xavier_initializer()
# He : tf_contrib.layers.variance_scaling_initializer()
# Normal : tf.random_normal_initializer(mean=0.0, stddev=0.02)
# Truncated_normal : tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
# Orthogonal : tf.orthogonal_initializer(1.0) / relu = sqrt(2), the others = 1.0

##################################################################################
# Regularizers
##################################################################################

def orthogonal_regularizer(scale) :
    """ Defining the Orthogonal regularizer and return the function at last to be used in Conv layer as kernel regularizer"""

    def ortho_reg(w) :
        """ Reshaping the matrxi in to 2D tensor for enforcing orthogonality"""
        c = w.get_shape().as_list()[-1]

        w = tf.reshape(w, [-1, c])

        """ Declaring a Identity Tensor of appropriate size"""
        identity = tf.eye(c)

        """ Regularizer Wt*W - I """
        w_transpose = tf.transpose(w)
        w_mul = tf.matmul(w_transpose, w)
        reg = tf.subtract(w_mul, identity)

        """Calculating the Loss Obtained"""
        ortho_loss = tf.nn.l2_loss(reg)

        return scale * ortho_loss

    return ortho_reg

def orthogonal_regularizer_fully(scale) :
    """ Defining the Orthogonal regularizer and return the function at last to be used in Fully Connected Layer """

    def ortho_reg_fully(w) :
        """ Reshaping the matrix in to 2D tensor for enforcing orthogonality"""
        c = w.get_shape().as_list()[-1]

        """Declaring a Identity Tensor of appropriate size"""
        identity = tf.eye(c)
        w_transpose = tf.transpose(w)
        w_mul = tf.matmul(w_transpose, w)
        reg = tf.subtract(w_mul, identity)

        """ Calculating the Loss """
        ortho_loss = tf.nn.l2_loss(reg)

        return scale * ortho_loss

    return ortho_reg_fully

##################################################################################
# Regularization
##################################################################################

# l2_decay : tf_contrib.layers.l2_regularizer(0.0001)
# orthogonal_regularizer : orthogonal_regularizer(0.0001) / orthogonal_regularizer_fully(0.0001)

weight_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
weight_regularizer = orthogonal_regularizer(0.0001)
weight_regularizer_fully = orthogonal_regularizer_fully(0.0001)

# Regularization only G in BigGAN

##################################################################################
# Layer
##################################################################################

# pad = ceil[ (kernel - stride) / 2 ]

def conv(x, channels, kernel=4, stride=2, pad=0, pad_type='zero', use_bias=True, sn=False, scope='conv_0'):
    with tf.variable_scope(scope):
        if pad > 0:
            h = x.get_shape().as_list()[1]
            if h % stride == 0:
                pad = pad * 2
            else:
                pad = max(kernel - (h % stride), 0)

            pad_top = pad // 2
            pad_bottom = pad - pad_top
            pad_left = pad // 2
            pad_right = pad - pad_left

            padding = [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]]

            if pad_type == 'zero' :
                x = tf.pad(x, padding)
            if pad_type == 'reflect' :
                x = tf.pad(x, padding, mode='REFLECT')

        if scope.__contains__('generator') :
            w = tf.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], channels], initializer=weight_init,
                                regularizer=weight_regularizer)
        else :
            w = tf.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], channels], initializer=weight_init,
                                regularizer=None)
            
        if sn:
            x = tf.nn.conv2d(input=x, filter=spectral_norm(w),
                             strides=[1, stride, stride, 1], padding='VALID')
        else:
            x = tf.nn.conv2d(input=x, filter=w,
                             strides=[1, stride, stride, 1], padding='VALID')
        
        if use_bias :
            bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
            x = tf.nn.bias_add(x, bias)

        return x


def deconv(x, channels, kernel=4, stride=2, padding='SAME', use_bias=True, sn=False, scope='deconv_0'):
    with tf.variable_scope(scope):
        x_shape = x.get_shape().as_list()

        if padding == 'SAME':
            output_shape = tf.stack([tf.shape(x)[0], x_shape[1] * stride, x_shape[2] * stride, channels])

        else:
            output_shape = tf.stack([tf.shape(x)[0], x_shape[1] * stride + max(kernel - stride, 0), x_shape[2] * stride + max(kernel - stride, 0), channels])

        
        w = tf.get_variable("kernel", shape=[kernel, kernel, channels, x_shape[-1]], initializer=weight_init, regularizer=weight_regularizer)
        
        if sn:
            x = tf.nn.conv2d_transpose(x, filter=spectral_norm(w), output_shape=output_shape, strides=[1, stride, stride, 1], padding=padding)
        else:
            x = tf.nn.conv2d_transpose(x, filter=w, output_shape=output_shape, strides=[1, stride, stride, 1], padding=padding)

        if use_bias :
            bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
            x = tf.nn.bias_add(x, bias)

        return x

def fully_connected(x, units, use_bias=True, is_training=True, sn=False, scope='fully_0'):
    with tf.variable_scope(scope):
        x = flatten(x)
        shape = x.get_shape().as_list()
        channels = shape[-1]

        
        if scope.__contains__('generator'):
            w = tf.get_variable("kernel", [channels, units], tf.float32, initializer=weight_init, 
            					regularizer=weight_regularizer_fully, trainable=is_training)
        else :
            w = tf.get_variable("kernel", [channels, units], tf.float32, initializer=weight_init,
            					 regularizer=None, trainable=is_training)      
        
        if sn:
        	x = tf.matmul(x, spectral_norm(w)) 
        else:
        	x = tf.matmul(x, w)
            
        if use_bias:
            bias = tf.get_variable("bias", [units], initializer=tf.constant_initializer(0.0), trainable=is_training)
            x = tf.nn.bias_add(x, bias)

        return x

def flatten(x) :
    return tf.keras.layers.Flatten()(x)

def hw_flatten(x) :
    _,h,w,c = x.get_shape().as_list()
    return tf.reshape(x, shape=[-1, h*w, c])

##################################################################################
# Residual-block, Self-Attention-block
##################################################################################

def resblock(x_init, channels, kernel=3, pads=[1,1], use_bias=True, is_training=True, sn=False, scope='resblock'):
    with tf.variable_scope(scope):
        with tf.variable_scope('res1'):
            x = conv(x_init, channels, kernel=kernel, stride=1, pad=pads[0], use_bias=use_bias, sn=sn)
            x = instance_norm(x, is_training)
            x = relu(x)

        with tf.variable_scope('res2'):
            x = conv(x, channels, kernel=kernel, stride=1, pad=pads[1], use_bias=use_bias, sn=sn)
            x = instance_norm(x, is_training)

        return relu(x + x_init)

def resblock_condition(x_init, conds, channels, kernel=3, pads=[1,1], use_bias=True, is_training=True, sn=False, scope='resblock'):
    with tf.variable_scope(scope):
        with tf.variable_scope('res1'):
            x = conv(x_init, channels, kernel=kernel, stride=1, pad=pads[0], use_bias=use_bias, sn=sn)
            x = adaptive_instance_norm(x, conds[:2], is_training)
            x = relu(x)

        with tf.variable_scope('res2'):
            x = conv(x, channels, kernel=kernel, stride=1, pad=pads[1], use_bias=use_bias, sn=sn)
            x = adaptive_instance_norm(x, conds[2:], is_training)

        return relu(x + x_init)

def resblock_up(x_init, channels, use_bias=True, is_training=True, sn=False, scope='resblock_up'):
    with tf.variable_scope(scope):
        with tf.variable_scope('res1'):
            x = instance_norm(x_init, is_training)
            x = relu(x)
            x = deconv(x, channels, kernel=3, stride=2, use_bias=use_bias, sn=sn)

        with tf.variable_scope('res2') :
            x = instance_norm(x, is_training)
            x = relu(x)
            x = deconv(x, channels, kernel=3, stride=1, use_bias=use_bias, sn=sn)

        with tf.variable_scope('skip') :
            x_skip = deconv(x_init, channels, kernel=3, stride=2, use_bias=use_bias, sn=sn)


    return x + x_skip

def resblock_up_condition(x_init, conds, channels, use_bias=True, is_training=True, sn=False, scope='resblock_up'):
    with tf.variable_scope(scope):
        with tf.variable_scope('res1'):
            x = adaptive_instance_norm(x_init, conds[:2], is_training)
            x = relu(x)
            x = deconv(x, channels, kernel=3, stride=2, use_bias=use_bias, sn=sn)

        with tf.variable_scope('res2') :
            x = adaptive_instance_norm(x, conds[2:], is_training)
            x = relu(x)
            x = deconv(x, channels, kernel=3, stride=1, use_bias=use_bias, sn=sn)

        with tf.variable_scope('skip') :
            x_skip = deconv(x_init, channels, kernel=3, stride=2, use_bias=use_bias, sn=sn)


    return x + x_skip


def resblock_down(x_init, channels, use_bias=True, is_training=True, sn=False, scope='resblock_down'):
    with tf.variable_scope(scope):
        with tf.variable_scope('res1'):
            x = instance_norm(x_init, is_training)
            x = relu(x)
            x = conv(x, channels, kernel=3, stride=2, pad=1, use_bias=use_bias, sn=sn)

        with tf.variable_scope('res2') :
            x = instance_norm(x, is_training)
            x = relu(x)
            x = conv(x, channels, kernel=3, stride=1, pad=1, use_bias=use_bias, sn=sn)

        with tf.variable_scope('skip') :
            x_skip = conv(x_init, channels, kernel=3, stride=2, pad=1, use_bias=use_bias, sn=sn)


    return x + x_skip

def resblock_down_no_instance_norm(x_init, channels, use_bias=True, is_training=True, sn=False, scope='resblock_down'):
    with tf.variable_scope(scope):
        with tf.variable_scope('res1'):
            x = relu(x_init)
            x = conv(x, channels, kernel=3, stride=2, pad=1, use_bias=use_bias, sn=sn)

        with tf.variable_scope('res2') :
            x = relu(x)
            x = conv(x, channels, kernel=3, stride=1, pad=1, use_bias=use_bias, sn=sn)

        with tf.variable_scope('skip') :
            x_skip = conv(x_init, channels, kernel=3, stride=2, pad=1, use_bias=use_bias, sn=sn)


    return x + x_skip

def self_attention(x, channels, sn=False, scope='self_attention'):
    with tf.variable_scope(scope):
        f = conv(x, channels // 8, kernel=1, stride=1, sn=sn, scope='f_conv')  # [bs, h, w, c']
        g = conv(x, channels // 8, kernel=1, stride=1, sn=sn, scope='g_conv')  # [bs, h, w, c']
        h = conv(x, channels, kernel=1, stride=1, sn=sn, scope='h_conv')  # [bs, h, w, c]

        # N = h * w
        s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)  # # [bs, N, N]

        beta = tf.nn.softmax(s)  # attention map

        # TODO: check that softmax along the last dimension was the correct one

        o = tf.matmul(beta, hw_flatten(h))  # [bs, N, C]
        gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))

        o = tf.reshape(o, shape=x.shape)  # [bs, h, w, C]
        x = gamma * o + x

    return x

def self_attention_2(x, channels, sn=False, scope='self_attention'):
    with tf.variable_scope(scope):
        f = conv(x, channels // 8, kernel=1, stride=1, sn=sn, scope='f_conv')  # [bs, h, w, c']
        f = max_pooling(f)

        g = conv(x, channels // 8, kernel=1, stride=1, sn=sn, scope='g_conv')  # [bs, h, w, c']

        h = conv(x, channels // 2, kernel=1, stride=1, sn=sn, scope='h_conv')  # [bs, h, w, c]
        h = max_pooling(h)

        # N = h * w
        s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)  # # [bs, N, N]

        beta = tf.nn.softmax(s)  # attention map

        # TODO: check that softmax along the last dimension was the correct one

        o = tf.matmul(beta, hw_flatten(h))  # [bs, N, C]
        gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))

        o = tf.reshape(o, shape=[-1, x.shape[1], x.shape[2], channels // 2])  # [bs, h, w, C]
        o = conv(o, channels, kernel=1, stride=1, sn=sn, scope='attn_conv')
        x = gamma * o + x

    return x

##################################################################################
# Sampling
##################################################################################

def global_avg_pooling(x):
    gap = tf.reduce_mean(x, axis=[1, 2])
    return gap

def global_sum_pooling(x) :
    gsp = tf.reduce_sum(x, axis=[1, 2])
    return gsp

def max_pooling(x) :
    x = tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    return x

def up_sample(x, scale_factor=2):
    _, h, w, _ = x.get_shape().as_list()
    new_size = [h * scale_factor, w * scale_factor]
    return tf.image.resize_nearest_neighbor(x, size=new_size)

##################################################################################
# Activation function
##################################################################################

def lrelu(x, alpha=0.2):
    return tf.nn.leaky_relu(x, alpha)


def relu(x):
    return tf.nn.relu(x)


def tanh(x):
    return tf.tanh(x)

def sigmoid(x):
    return tf.nn.sigmoid(x)

##################################################################################
# Normalization function
##################################################################################

def instance_norm(x, is_training=True, scope='IN'):
    # TODO: replace with tf.nn.batch_normalization(x, batch_mean, batch_var, beta, gamma, epsilon)
    with tf.variable_scope(scope) :
        _, _, _, c = x.get_shape().as_list()
        decay = 0.9
        epsilon = 1e-05

        test_mean = tf.get_variable("pop_mean", shape=[c], dtype=tf.float32, initializer=tf.constant_initializer(0.0), trainable=False)
        test_var  = tf.get_variable("pop_var",  shape=[c], dtype=tf.float32, initializer=tf.constant_initializer(1.0), trainable=False)
        
        beta  = tf.get_variable("beta", shape=[c], dtype=tf.float32, initializer=tf.constant_initializer(0.0), trainable=is_training)
        gamma = tf.get_variable("gamma", shape=[c], dtype=tf.float32, initializer=tf.constant_initializer(0.0), trainable=is_training)
               
        if is_training:
            # Update exponential moving averages of the batch mean and var
            batch_mean, batch_var = tf.nn.moments(x, [1, 2])
            
            ema_mean = tf.assign(test_mean, test_mean * decay + batch_mean[-1] * (1 - decay))
            ema_var  = tf.assign(test_var,  test_var  * decay + batch_var[-1]  * (1 - decay))

            with tf.control_dependencies([ema_mean, ema_var]):
                return tf.nn.batch_normalization(x, batch_mean[-1], batch_var[-1], beta, gamma, epsilon)
        else:
            return tf.nn.batch_normalization(x, test_mean, test_var, beta, gamma, epsilon)

def adaptive_instance_norm(x, z, is_training=True, scope='AdaIN'):
    with tf.variable_scope(scope) :
        _, _, _, c = x.get_shape().as_list()
        decay = 0.9
        epsilon = 1e-05

        test_mean = tf.get_variable("pop_mean", shape=[c], dtype=tf.float32, initializer=tf.constant_initializer(0.0), trainable=False)
        test_var  = tf.get_variable("pop_var",  shape=[c], dtype=tf.float32, initializer=tf.constant_initializer(1.0), trainable=False)

        beta = z[0]
        gamma = z[1]
        
        if is_training:
            # Update exponential moving averages of the batch mean and var
            batch_mean, batch_var = tf.nn.moments(x, [1, 2])
            ema_mean = tf.assign(test_mean, test_mean * decay + batch_mean[-1] * (1 - decay))
            ema_var  = tf.assign(test_var,  test_var  * decay + batch_var[-1]  * (1 - decay))

            with tf.control_dependencies([ema_mean, ema_var]):
                return tf.nn.batch_normalization(x, batch_mean[-1], batch_var[-1], beta, gamma, epsilon)
        else:
            return tf.nn.batch_normalization(x, test_mean, test_var, beta, gamma, epsilon)


def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """

        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm