import numpy as np
import os
import tensorflow as tf
import logging
from tensorflow.python.framework import dtypes
from experiment_manage.util import build_print_shape  # test_run
from experiment_manage.core import Builder, BuilderFactory, DTYPE


# noinspection PyPep8Naming
def partial_reduce_mean(T, part_reduce):
    """
    Summarise an input tensor along the last axis.
    :param T: input numpy tensor
    :param part_reduce: a list, each element is a list of indices: how to group
      slices of T into one slice of output S
    :return:

    For example, if T is of [3 x 4 x 5], part_reduce=[[0,1], [2,3,4]], the output S
    will be a tensor of [3 x 4 x 2]. S[:,:,0] = mean(T[:,:,(0,1)]),
    S[:,:,1] = mean(T[:,:,(2,3,4)]),
    """
    S = np.zeros(T.shape[:-1] + (len(part_reduce),))
    for n, ii in enumerate(part_reduce):
        for i in ii:
            S[..., n] += T[..., i]
        S[..., n] /= float(len(ii))
    return S


# noinspection PyPep8Naming
def index_group(N0, N1):
    assert N1 < N0
    sep = np.linspace(0, N0, num=N1 + 1).astype(np.int)
    return [list(range(i0, i1)) for i0, i1 in zip(sep[:-1], sep[1:])]


# noinspection PyAttributeOutsideInit,PyMethodMayBeStatic,PyPep8Naming
class FullConvNet_VGG16(Builder):
    def __init__(self, conf):
        """
        :param conf: experiment configure
            - encoder: options related to encoder network
            - solver: options related to training of the model, such as weight-decay
        """
        super(FullConvNet_VGG16, self).__init__(conf)
        self.num_classes = conf['data']['num_classes']
        self.fc_shape_convert = conf['encoder']['fc_to_conv']

        self.debug = conf['debug']
        if self.debug['load_weights']:
            print "VGG params loaded, fields: "
            for k in self.vgg_params.keys():
                print "\t{}:".format(k)
                for wi, w in enumerate(self.vgg_params[k]):
                    print "\t\t[{}]: {}".format(wi, w.shape)
        self.conf = conf

        # phrase related
        self.phrase = None
        self.weight_decay_rate = 0.0
        self.dropout_rate = 0.0
        self.vgg_params = None

    @staticmethod
    def load_pre_trained_vgg_weights(filename, channel_means):
        logging.debug("Load from {}".format(filename))
        vgg_params = np.load(filename).item()
        assert isinstance(vgg_params, dict)
        vgg_params['mean_r'] = channel_means['r']
        vgg_params['mean_g'] = channel_means['g']
        vgg_params['mean_b'] = channel_means['b']
        return vgg_params

    def _get_tf_variable(self, **kwargs):
        return tf.get_variable(dtype=DTYPE, trainable=(self.phrase == 'train'), **kwargs)

    def build(self, inputs, phrase):
        """
        :param inputs, one-element list = [rgb_batch], which is a image batch
            tensor, [ None x height x width x num_channels ], The shape is
            how images are loaded.
        :param phrase: train or infer
        """
        self.phrase = phrase
        if phrase == 'train':
            self.dropout_rate = self.conf['solver']['dropout_rate']

        self.weight_decay_rate = self.conf['objective']['weight_decay']
        param_file = os.path.join(self.conf['path']['base'],
                                  self.conf['encoder']['pre_trained_param'])
        logging.debug("pm:{} / {} / {}".format(self.conf['path']['base'],self.conf['encoder']['pre_trained_param'], param_file))
        self.vgg_params = self.load_pre_trained_vgg_weights(
            filename=param_file,
            channel_means=self.conf['encoder']['channel_means'])
        logging.info("Based on pre-trained VGG (cold start)"
                     "\n\t{}".format(param_file))

        rgb_batch = inputs[0]
        assert isinstance(rgb_batch, tf.Tensor)
        assert rgb_batch.get_shape().ndims == 4
        assert rgb_batch.get_shape()[3] == 3  # NxHxWxC, 3 channels, rgb

        with tf.name_scope("Processing"):
            rgb_batch = tf.cast(rgb_batch, dtype=dtypes.float32)
            ch_r, ch_g, ch_b = tf.split(rgb_batch, 3, axis=3)
            bgr_batch = tf.concat([
                ch_b - self.vgg_params['mean_b'],
                ch_g - self.vgg_params['mean_g'],
                ch_r - self.vgg_params['mean_r']], axis=3)
            if self.debug['input']:
                bgr_batch = build_print_shape(bgr_batch, "BGR Image", first_n=1)

        # VGG convolutional
        self.conv1_1 = self._conv_layer(bgr_batch, "conv1_1", False)
        self.conv1_2 = self._conv_layer(self.conv1_1, "conv1_2", False)
        self.pool1 = self._max_pool(self.conv1_2, "pool1", self.debug['conv'])

        self.conv2_1 = self._conv_layer(self.pool1, "conv2_1", False)
        self.conv2_2 = self._conv_layer(self.conv2_1, "conv2_2", False)
        self.pool2 = self._max_pool(self.conv2_2, "pool2", self.debug['conv'])

        self.conv3_1 = self._conv_layer(self.pool2, "conv3_1", False)
        self.conv3_2 = self._conv_layer(self.conv3_1, "conv3_2", False)
        self.conv3_3 = self._conv_layer(self.conv3_2, "conv3_3", False)
        self.pool3 = self._max_pool(self.conv3_3, "pool3", self.debug['conv'])

        self.conv4_1 = self._conv_layer(self.pool3, "conv4_1", False)
        self.conv4_2 = self._conv_layer(self.conv4_1, "conv4_2", False)
        self.conv4_3 = self._conv_layer(self.conv4_2, "conv4_3", False)
        self.pool4 = self._max_pool(self.conv4_3, "pool4", self.debug['conv'])

        self.conv5_1 = self._conv_layer(self.pool4, "conv5_1", False)
        self.conv5_2 = self._conv_layer(self.conv5_1, "conv5_2", False)
        self.conv5_3 = self._conv_layer(self.conv5_2, "conv5_3", False)
        self.pool5 = self._max_pool(self.conv5_3, "pool5", self.debug['conv'])

        self.fc6 = self._fc_layer(self.pool5, "fc6",
                                  shape_convert=self.fc_shape_convert['fc6'],
                                  do_relu=True, debug=self.debug['fc'])
        if self.phrase == 'train':
            self.fc6 = tf.nn.dropout(self.fc6, self.dropout_rate)

        self.fc7 = self._fc_layer(self.fc6, "fc7",
                                  shape_convert=self.fc_shape_convert['fc7'],
                                  do_relu=True, debug=self.debug['fc'])
        if self.phrase == 'train':
            self.fc7 = tf.nn.dropout(self.fc7, self.dropout_rate)

        self.fc8 = self._adapt_fc_layer(self.fc7, "fc8",
                                        shape_convert=self.fc_shape_convert['fc8'],
                                        do_relu=False,
                                        num_classes=self.num_classes,
                                        debug=self.debug['fc'])

        pool4_shape = tf.shape(self.pool4)
        self.upscore2 = self._upscore_layer(self.fc8, "upscore2",
                                            ksize=4, stride=2,
                                            num_classes=self.num_classes,
                                            up_w=pool4_shape[2],
                                            up_h=pool4_shape[1],
                                            debug=self.debug['up'])

        self.score_pool4 = self._score_layer(self.pool4, "score_pool4",
                                             num_classes=self.num_classes,
                                             random_weight_stddev=0.001)

        self.fuse_pool4 = tf.add(self.upscore2, self.score_pool4)

        input_shape = tf.shape(bgr_batch)
        self.upscore32 = self._upscore_layer(self.fuse_pool4, "upscore32",
                                             ksize=32, stride=16,
                                             num_classes=self.num_classes,
                                             up_w=input_shape[2], up_h=input_shape[1],
                                             debug=self.debug['up'])

        return [self.upscore32, ]

    def _score_layer(self, bottom, lname, num_classes,
                     random_weight_stddev=None):
        """
        :brief linearly converts the output of a convolutional layer to class
          prediction (in terms of pre-active likelihood). I.e. if you have a
          set of "convolutional features", e.g. [. x H x W x C], and want to
          use them for predicting pixel-wise class-likelihood, i.e. the output
          is [. x H x W x #.classes], this layer provides linear combination
          of the C channels into #.classes

        :param bottom, the input tensor of [. x H x W x C]
        :param num_classes, see brief above.
        :return [. x H x W x #.classes] tensor
        """
        with tf.variable_scope(lname) as scope:
            num_infeat = bottom.get_shape()[3].value
            w_shape = [1, 1, num_infeat, num_classes]
            if random_weight_stddev is None:
                stddev = (2 / num_infeat) ** 0.5
            else:
                stddev = random_weight_stddev
            init = tf.truncated_normal_initializer(stddev=stddev)
            filt = self._get_tf_variable(
                name='weights', initializer=init, shape=w_shape)
            if not scope.reuse:
                # add to loss
                wdl = tf.multiply(tf.nn.l2_loss(filt), self.weight_decay_rate)
                tf.add_to_collection('losses', wdl)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            bias = self._get_tf_variable(
                name='biases', shape=[num_classes],
                initializer=tf.constant_initializer(0.0))
            top = tf.nn.bias_add(conv, bias)

        return top

    def _upscore_layer(self, bottom, lname, ksize, stride,
                       num_classes,
                       up_w=None, up_h=None,
                       debug=False):
        with tf.variable_scope(lname):
            # determin output shape
            true_bottom_shape = tf.shape(bottom)
            num_imgs = true_bottom_shape[0]
            w = true_bottom_shape[2]
            h = true_bottom_shape[1]
            if up_w is None:
                up_w = stride * (w - 1) + 1
            if up_h is None:
                up_h = stride * (h - 1) + 1
            upscore_shape = tf.stack([num_imgs, up_h, up_w, num_classes])

            num_in_channels = bottom.get_shape()[3]
            assert num_in_channels == num_classes
            filt = self.get_deconv_filter(ksize, num_classes)
            upscore = tf.nn.conv2d_transpose(
                bottom, filt, upscore_shape,
                strides=[1, stride, stride, 1],
                padding='SAME')
            if debug:
                upscore = build_print_shape(upscore, msg="upscore {}".format(lname))
        return upscore

    def _conv_layer(self, bottom, lname, debug=False):
        """
        :param bottom is the input. It can be a minibatch of images or
          the output of previous layers.
        :param lname, conv.layer name, see @reload_conv_filter
        :return activation of the conv.layer
        """
        # variable scope is necessary, the variable will have a name like "weights"
        # for all layers.
        with tf.variable_scope(lname):
            filt = self.reload_conv_filter(lname)
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            conv_bias = self.reload_internal_bias(lname)
            bias = tf.nn.bias_add(conv, conv_bias)
            relu = tf.nn.relu(bias)
            if debug:
                relu = build_print_shape(relu, "conv {}:".format(lname), 1)
            return relu

    def _max_pool(self, bottom, slname, debug):
        """
        :param slname name of the "superlayer", see @reload_conv_filter
        """
        pool = tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                              padding='SAME',
                              name=slname)
        if debug:
            pool = build_print_shape(pool, "maxpool {}:".format(slname), 1)
        return pool

    def _fc_layer(self, bottom, lname, shape_convert,
                  do_relu, debug, num_classes=None):
        """
        :brief _fc_layer build a fully-connected layer up the @bottom, using
          convolution operation, e.g. if in the traditional net, the bottom
          has 7 x 7 x Cin, the output has Cout channels, a fully connected
          net will need (7x7xCin) "flattened" x Cout weights to specify.
          Now we re-shape the 7x7xCin as a convolution on a 7x7 cell, with
          Cin in-channels and Cout out-channels

          Adapt trained model to new tasks: If desired out-channel number
          Cout does not match pretrained filter, generally, much less than
          the pretrained filter, VGG was trained on image-net, with 1000
          classes to predict, the loader will combine the output weights
          of multiple (original output) channels into one (new) output
          channel.
        :shape_convert
        - fc_weight_shape, the shape of original VGG's fully connected layer,
          for confirmation purpose only
        - conv_kernel_shape, the kernel weights of the newly constructed convolution
          layer.
        :param num_classes, desired number of output channels.
        """
        kshape = shape_convert['conv_kernel_shape']
        wshape = shape_convert['fc_weight_shape']
        with tf.variable_scope(lname):
            kweights_var = self.reload_fc_filter(lname, kshape, wshape)
            conv = tf.nn.conv2d(bottom, kweights_var, [1, 1, 1, 1], padding='SAME')
            bias_var = self.adapt_pred_bias(lname, num_classes)
            fc = tf.nn.bias_add(conv, bias_var)
            if do_relu:
                fc = tf.nn.relu(fc)
            if debug:
                fc = build_print_shape(fc, "fc {}:".format(lname))
        return fc

    def _adapt_fc_layer(self, bottom, lname, shape_convert,
                        do_relu, debug, num_classes=None):
        """
        Partially reuse a pre-trained fully connected (by convolution) layer, with
        fewer output classes. Specifically,  VGG was trained on image-net, with 1000
        classes to predict. But usually, we have much fewer classes to predict. So
        we combine the weights for multiple original classes to one target class.
        See also @adapt_bias

        :param num_classes, desired number of output channels.
        """

        kshape = shape_convert['conv_kernel_shape']
        wshape = shape_convert['fc_weight_shape']
        with tf.variable_scope(lname):
            kweights_var = self.adapt_fc_filter(lname, kshape, num_classes, wshape)
            bias_var = self.adapt_pred_bias(lname, num_classes)
            conv = tf.nn.conv2d(bottom, kweights_var, [1, 1, 1, 1], padding='SAME')
            fc = tf.nn.bias_add(conv, bias_var)
            if do_relu:
                fc = tf.nn.relu(fc)
            if debug:
                fc = build_print_shape(fc, "fc {}:".format(lname))
        return fc

    def reload_fc_filter(self, lname, kshape, wshape=None):
        if wshape:
            assert self.vgg_params[lname][0].shape == tuple(wshape)
        w = self.vgg_params[lname][0].reshape(kshape)
        init = tf.constant_initializer(value=w, dtype=tf.float32)
        w_var = self._get_tf_variable(name='fc_wt', initializer=init, shape=kshape)
        if not tf.get_variable_scope().reuse:
            wd = tf.multiply(tf.nn.l2_loss(w_var),
                             self.weight_decay_rate,
                             name="weight_decay_loss")
            tf.add_to_collection("losses", wd)
        return w_var

    def adapt_fc_filter(self, lname, kshape_ori, num_classes, wshape=None):
        """
        Prepare adapted weights for new number of outputs. See also @_adapt_fc_layer
        :param lname:
        :param kshape_ori: all original weights **as a conv kernel**
        :param num_classes: how many new classes
        :param wshape: the weights shape in pre-trained vgg
        :return:
        """
        if wshape:
            assert tuple(wshape) == self.vgg_params[lname][0].shape
        w = self.vgg_params[lname][0].reshape(kshape_ori)
        if (num_classes is None) or num_classes == kshape_ori[3]:
            return self.reload_fc_filter(lname, kshape_ori)

        C0 = kshape_ori[3]
        assert num_classes < C0
        w_new = partial_reduce_mean(w, index_group(C0, num_classes))
        init = tf.constant_initializer(value=w_new, dtype=tf.float32)
        kshape_new = w_new.shape
        w_var = self._get_tf_variable(name='fc_wt', initializer=init, shape=kshape_new)

        if not tf.get_variable_scope().reuse:
            wd = tf.multiply(tf.nn.l2_loss(w_var),
                             self.weight_decay_rate,
                             name="weight_decay_loss")
            tf.add_to_collection("losses", wd)
        return w_var

    def get_deconv_filter_value_ref(self, l):
        """
        Reference
        """
        f = np.ceil(l / 2.0)
        c = (2 * f - 1 - f % 2) / (2.0 * f)
        # print "Deconv Filter of {}x{}".format(l,l)
        # print "f={}, c={}".format(f,c)
        bilinear = np.zeros([l, l])
        for x in range(l):
            for y in range(l):
                value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
                bilinear[x, y] = value
                # if x==0:
                #  print "y={}, y/f={}, |y/f-c|={}, 1-|.|={}".format(
                #    y, y/f, abs(y/f-c), 1-abs(y/f-c))

                # print "x={}, x/f={}, |x/f-c|={}, 1-|.|={}".format(
                #  x, x/f, abs(x/f-c), 1-abs(x/f-c))
        return bilinear

    #   def get_deconv_filter_bilinear_interp(self, l):
    #     """
    #
    #     (0,0) ... (l-1, 0)
    #     [0,0]      [1,0]
    #     .
    #     .
    #     .
    #     (0,l-1) ... (l-1,l-1)
    #     [0,1]        [1,1]
    #     """
    #     k2t = np.linspace(0,2,l+2)
    #
    #     bilinear=np.zeros((l,l))
    #     fw=lambda x: 1 - abs(x-1)
    #     for x in range(1,l+1):
    #       for y in range(1,l+1):
    #         bilinear[y-1,x-1] = fw(k2t[y])*fw(k2t[x])
    #     return bilinear
    #     #weights = np.zeros(f_shape)
    #     #for i in range(f_shape[2]):
    #     #    weights[:, :, i, i] = bilinear
    #
    #     #init = tf.constant_initializer(value=weights,
    #     #                               dtype=tf.float32)
    #     #return tf.get_variable(name="up_filter", initializer=init,
    #     #                       shape=weights.shape)
    #
    # noinspection PyPep8Naming
    def get_deconv_filter(self, ksize, C):
        """
        Make a filter for bilinear interp
        :param ksize: kernel size
        :param C: number of in-out channels (classes), in-out must match
        """
        # bilinear_interp=self.get_deconv_filter_bilinear_interp(ksize)
        bilinear_interp = self.get_deconv_filter_value_ref(ksize)
        f_val = np.zeros([ksize, ksize, C, C])
        for i in range(C):
            f_val[:, :, i, i] = bilinear_interp[...]
        init = tf.constant_initializer(value=f_val,
                                       dtype=tf.float32)
        return self._get_tf_variable(name='up_filter',
                                     initializer=init,
                                     shape=f_val.shape)

    def reload_conv_filter(self, lname):
        """
        :param lname represents a convolutional "layer" name. A "layer" is a
          notion for several computations related to filtering the input with
          convolutional kernels, adding bias to the results and sending the
          L2-norm of the conv-kernel to the loss. NOT to be confused with the
          "super-layer", collecting several convolutional layers and a
          pooling layer.

        :return a tensor, the convolutional kernel
        """
        w = self.vgg_params[lname][0]  # by convention, vgg-params['conv2_1']
        # has two items: [0] is the conv-kernel and [1] is the bias

        init = tf.constant_initializer(value=w, dtype=tf.float32)

        w_var = self._get_tf_variable(name='conv_filter',
                                      initializer=init,
                                      shape=w.shape)

        if not tf.get_variable_scope().reuse:
            # first time, put into weight decay
            wd = tf.multiply(tf.nn.l2_loss(w_var),
                             self.weight_decay_rate,
                             name="weight_decay_loss")
            tf.add_to_collection("losses", wd)

        if self.debug['load_conv_weights']:
            print "-- Initializing VGG conv layer [{}] of shape {}".format(
                w_var.name, w_var.get_shape())

        return w_var

    def reload_internal_bias(self, lname):
        """
        Create variables for bias of internal layers.
        :param lname:
        :return: b_var, tensorflow variable of the bias
        """
        b = self.vgg_params[lname][1]
        init = tf.constant_initializer(value=b, dtype=tf.float32)
        b_var = self._get_tf_variable(name='bias', initializer=init, shape=[b.size, ])
        return b_var

    def adapt_pred_bias(self, lname, num_classes):
        """
        Create variables for bias of output layers, i.e. the original VGG
        predicts among 1000 object classes, while the new application may
        only need, say 15, classes. Parameters for the old classes will
        be arranged to the new classes.
        TODO: I don't think summarise is much superior to simplier methods,
        e.g. directly take the first few biases or setting up to zero.
        :param lname: layer name
        :param num_classes: number of new classes
        :return:
        """

        b = self.vgg_params[lname][1]
        num_orig_classes = b.size
        if num_classes == num_orig_classes or num_classes is None:
            return self.reload_internal_bias(lname)

        assert num_classes < num_orig_classes

        # summarise
        num_old_classes_per_new = num_orig_classes // num_classes

        b_new = np.zeros([num_classes, ])
        for j in range(0, num_orig_classes, num_old_classes_per_new):
            j_ = j + num_old_classes_per_new
            i_new = j // num_old_classes_per_new
            b_new[i_new] = np.mean(b[j:j_])
        init = tf.constant_initializer(value=b_new, dtype=tf.float32)
        bshape_new = [num_classes, ]

        b_var = self._get_tf_variable(name='bias',
                                      initializer=init,
                                      shape=bshape_new)
        return b_var


class EncoderBuilderFactory(BuilderFactory):
    """
    Wrapper class of building encoding network. NOTE this interface is useful of
    decoupling specific networks from higher-level construction. I.e.,
    though now we only have VGG16 FCN, this interface allows us to choose from
    all available nets.

    EncoderBuilder-Wrapper is actually a factory.
    """

    def __init__(self):
        super(EncoderBuilderFactory, self).__init__()

    def get_builder(self, conf):
        return FullConvNet_VGG16(conf)


def id_str():
    return "Architecture construction:", __file__

    # TODO: add a __main__ for testing.
