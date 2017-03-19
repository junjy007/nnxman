import numpy as np
import tensorflow as tf


VGG_MEAN_B, VGG_MEAN_G, VGG_MEAN_R = (103.399, 116.779, 123.68)
class FullConvNet_VGG16:
    def __init__(self, vgg16file, opts):
        self.vgg_params=np.load(vgg16file).item()
        if opts['debug_info']['load_weights']:
            print "VGG params loaded, fields: "
            for k in self.vgg_params.keys():
                print "\t{}:".format(k)
                for wi,w in enumerate(self.vgg_params[k]):
                    print "\t\t[{}]: {}".format(wi, w.shape)
        self.weight_decay_rate=opts['weight_decay_rate']
        self.t_params={} # for easy debugging, NOT for saving/loading
        self.rng = np.random.RandomState(0)

    def build(self, rgb_input, train=False, 
            num_classes=20, random_init_fc8=False,
            debug=False):

        """
        :param rgb_input is a image batch tensor, [ None x height x width x
               num_channels ], The shape is how images are loaded.

        """
        with tf.name_scope("Processing"):
            ch_r, ch_g, ch_b = tf.split(rgb_input, 3, axis=3)
            bgr = tf.concat([
                ch_b - VGG_MEAN_B,
                ch_g - VGG_MEAN_G,
                ch_r - VGG_MEAN_R ], axis=3)

            if debug:
                bgr = tf.Print(bgr, [tf.shape(bgr)],
                        message="Shape of input image: ",
                        summarize=4, first_n=1)

        # VGG convolutional
        do_debug=True
        self.conv1_1 = self._conv_layer(bgr, "conv1_1", False)
        self.conv1_2 = self._conv_layer(self.conv1_1, "conv1_2", False)
        self.pool1 = self._max_pool(self.conv1_2, "pool1", do_debug)

        self.conv2_1 = self._conv_layer(self.pool1, "conv2_1", False)
        self.conv2_2 = self._conv_layer(self.conv2_1, "conv2_2", False)
        self.pool2 = self._max_pool(self.conv2_2, "pool2", do_debug)

        self.conv3_1 = self._conv_layer(self.pool2, "conv3_1", False)
        self.conv3_2 = self._conv_layer(self.conv3_1, "conv3_2", False)
        self.conv3_3 = self._conv_layer(self.conv3_2, "conv3_3", False)
        self.pool3 = self._max_pool(self.conv3_3, "pool3", do_debug)

        self.conv4_1 = self._conv_layer(self.pool3, "conv4_1", False)
        self.conv4_2 = self._conv_layer(self.conv4_1, "conv4_2", False)
        self.conv4_3 = self._conv_layer(self.conv4_2, "conv4_3", False)
        self.pool4 = self._max_pool(self.conv4_3, "pool4", do_debug)

        self.conv5_1 = self._conv_layer(self.pool4, "conv5_1", False)
        self.conv5_2 = self._conv_layer(self.conv5_1, "conv5_2", False)
        self.conv5_3 = self._conv_layer(self.conv5_2, "conv5_3", False)
        self.pool5 = self._max_pool(self.conv5_3, "pool5", do_debug)

        self.fc6 = self._fc_layer(self.pool5, "fc6", 
                do_relu=True, debug=do_debug)
        if train:
            self.fc6=tf.dropout(self.fc6,0.5)
        self.fc7 = self._fc_layer(self.fc6, "fc7",
                do_relu=True, debug=do_debug)
        if train:
            self.fc7=tf.dropout(self.fc7,0.5)

        self.fc8=self._fc_layer(self.fc7, "fc8", do_relu=False,
                                num_classes=num_classes,
                                debug=do_debug)

        pool4_shape=tf.shape(self.pool4)
        self.upscore2=self._upscore_layer(self.fc8, "upscore2",
                ksize=4, stride=2,
                num_classes=num_classes,
                up_w=pool4_shape[2],
                up_h=pool4_shape[1],
                debug=do_debug)

        self.score_pool4=self._score_layer(self.pool4, "score_pool4",
                num_classes=num_classes,
                random_weight_stddev=0.001)

        self.fuse_pool4 = tf.add(self.upscore2, self.score_pool4)

        input_shape=tf.shape(bgr)
        self.upscore32=self._upscore_layer(self.fuse_pool4, "upscore32",
                                           ksize=32, stride=16,
                                           num_classes=num_classes,
                                           up_w=input_shape[2],up_h=input_shape[1],
                                           debug=do_debug)
        
        return

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
            w_shape = [1,1,num_infeat, num_classes]
            if random_weight_stddev is None:
                stddev=(2/num_infeat)**0.5
            else:
                stddev=random_weight_stddev

            # TODO: why random seed results in different results in my net and reference net?
            # filt = tf.Print(filt, [filt[0,0,0]], message="Init filt: ", summarize=5, first_n=1)
            # -- DEBUG --
            DBG_FLAG = True
            if DBG_FLAG:
                tmp_w = np.load ('refnet_score_pool4_weights.npy')
                init = tf.constant_initializer(value=tmp_w, dtype=tf.float32)
                filt = tf.get_variable('weights', initializer=init, shape=w_shape)
            else:
                init = tf.truncated_normal_initializer(stddev=stddev)
                filt = tf.get_variable('weights', initializer=init,
                                       shape=w_shape)


            if (not scope.reuse):
                # add to loss
                wdl=tf.multiply(tf.nn.l2_loss(filt), self.weight_decay_rate)
                tf.add_to_collection('losses', wdl)
           
            conv = tf.nn.conv2d(bottom, filt, [1,1,1,1], padding='SAME')

            bias = tf.get_variable('biases', shape=[num_classes],
                    initializer=tf.constant_initializer(0.0))
            top = tf.nn.bias_add(conv, bias)

        self.t_params['lname']={'weights':filt, 'bias':bias}

        return top

    def _upscore_layer(self, bottom, lname, ksize, stride,
            num_classes,
            up_w=None, up_h=None,
            debug=False):
        with tf.variable_scope(lname) as scope:
            # determin output shape
            true_bottom_shape=tf.shape(bottom)
            num_imgs = true_bottom_shape[0]
            w = true_bottom_shape[2]
            h = true_bottom_shape[1]
            if up_w is None:
                up_w = stride*(w-1)+1

            # We should assert given up_w here, but it is a dynmaic shape
            # in tensorflow.
            #else:
            #    assert stride*(w-1)+1 <= up_w \
            #       and up_w <= stride*w
            if up_h is None:
                up_h = stride*(h-1)+1
            #else:
            #    assert stride*(h-1)+1 <= up_h \
            #       and up_h <= stride*h
            upscore_shape=tf.stack([num_imgs, up_h, up_w, num_classes])

            num_in_channels = bottom.get_shape()[3] 
            assert num_in_channels == num_classes
            filt = self.get_deconv_filter(ksize,num_classes)
            upscore = tf.nn.conv2d_transpose(
                    bottom, filt, upscore_shape,
                    strides=[1,stride,stride,1],
                    padding='SAME')
            if debug:
                upscore=tf.Print(upscore, [tf.shape(upscore)],
                        message="upscore layer {} shape: ".format(lname),
                        summarize=4, first_n=1)

            
        return upscore

    def _conv_layer(self, bottom, lname, debug=False):
        """
        :param bottom is the input. It can be a minibatch of images or
          the output of previous layers.
        :param lname, conv.layer name, see @get_conv_filter
        :return activation of the conv.layer
        """
        with tf.variable_scope(lname) as scope:
            filt = self.get_conv_filter(lname)
            conv = tf.nn.conv2d(bottom, filt, [1,1,1,1], padding='SAME')

            conv_bias = self.get_bias(lname)
            bias = tf.nn.bias_add(conv, conv_bias)
            relu = tf.nn.relu(bias)
            if debug:
                relu=tf.Print(relu,[tf.shape(relu)],
                        message="conv-layer {} output shape: ",
                        summarize=4,
                        first_n=1)
            return relu
    
    def _max_pool(self, bottom, slname, debug): 
        """
        :param slname name of the "superlayer", see @get_conv_filter
        """
        pool=tf.nn.max_pool(bottom, ksize=[1,2,2,1], strides=[1,2,2,1],
                padding='SAME',
                name=slname)
        if debug:
            pool=tf.Print(pool, [tf.shape(pool)],
                    message="Layer [{}] after pooling: ".format(slname),
                    summarize=4, first_n=1)
        return pool

    def _fc_layer(self, bottom, lname, do_relu, debug, num_classes=None):
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

        :param num_classes, desired number of output channels.
        """
        vgg_fc_shapes={'fc6':[7,7,512,4096],
                       'fc7':[1,1,4096,4096],
                       'fc8':[1,1,4096,1000]}
        kshape=vgg_fc_shapes[lname]
        with tf.variable_scope(lname) as scope:
            kweights_var=self.get_fc_filter(lname,kshape,num_classes)
            # debugging tmp begin
            kweights_var=tf.Print(kweights_var, [tf.shape(kweights_var)], 
                          message="#### fc-conv weights shape: ",
                          summarize=4, first_n=1)
            # debugging tmp end
            conv=tf.nn.conv2d(bottom, kweights_var, [1,1,1,1],
                              padding='SAME')
            bias_var=self.get_bias(lname,num_classes)
            fc=tf.nn.bias_add(conv, bias_var)
            if do_relu:
                fc=tf.nn.relu(fc)
            if debug:
                fc=tf.Print(fc,[tf.shape(fc)], 
                            message="fc layer {} shape: ".format(lname),
                            summarize=4, first_n=1)
        return fc
        
    def get_deconv_filter_value_ref(self, l):
        """
        Reference
        """
        f = np.ceil(l/2.0)
        c = (2 * f - 1 - f % 2) / (2.0 * f)
        print "Deconv Filter of {}x{}".format(l,l)
        print "f={}, c={}".format(f,c)
        bilinear = np.zeros([l,l])
        for x in range(l):
            for y in range(l):
                value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
                bilinear[x, y] = value
                if x==0:
                    print "y={}, y/f={}, |y/f-c|={}, 1-|.|={}".format(
                            y, y/f, abs(y/f-c), 1-abs(y/f-c))

            print "x={}, x/f={}, |x/f-c|={}, 1-|.|={}".format(
                    x, x/f, abs(x/f-c), 1-abs(x/f-c))
        return bilinear

    def get_deconv_filter_bilinear_interp(self, l):
        """

        (0,0) ... (l-1, 0)
        [0,0]      [1,0]
        .
        .
        .
        (0,l-1) ... (l-1,l-1)
        [0,1]        [1,1]
        """
        k2t = np.linspace(0,2,l+2)

        bilinear=np.zeros((l,l))
        fw=lambda x: 1 - abs(x-1)
        for x in range(1,l+1): 
            for y in range(1,l+1):
                bilinear[y-1,x-1] = fw(k2t[y])*fw(k2t[x])
        return bilinear
        #weights = np.zeros(f_shape)
        #for i in range(f_shape[2]):
        #    weights[:, :, i, i] = bilinear

        #init = tf.constant_initializer(value=weights,
        #                               dtype=tf.float32)
        #return tf.get_variable(name="up_filter", initializer=init,
        #                       shape=weights.shape)

    def get_deconv_filter(self, ksize, C):
        """
        Make a filter for bilinear interp
        :param ksize: kernel size
        :param C: number of in-out channels (classes), in-out must match
        """
        #bilinear_interp=self.get_deconv_filter_bilinear_interp(ksize)
        bilinear_interp=self.get_deconv_filter_value_ref(ksize)
        f_val=np.zeros([ksize,ksize,C,C])
        for i in range(C):
            f_val[:,:,i,i]=bilinear_interp[...]
        init = tf.constant_initializer(value=f_val,
                dtype=tf.float32)
        return tf.get_variable(name='up_filter',
                initializer=init,
                shape=f_val.shape)



    def get_conv_filter(self, lname):
        """
        :param lname represents a convolutional "layer" name. A "layer" is a
          notion for several computations related to filtering the input with
          convolutional kernels, adding bias to the results and sending the
          L2-norm of the conv-kernel to the loss. NOT to be confused with the
          "super-layer", collecting several convolutional layers and a
          pooling layer.

        :return a tensor, the convolutional kernel
        """
        w = self.vgg_params[lname][0] # by convention, vgg-params['conv2_1']
        # has two items: [0] is the conv-kernel and [1] is the bias

        init = tf.constant_initializer(value=w, dtype=tf.float32)

        w_var = tf.get_variable(name='conv_filter',
                                initializer=init,
                                shape=w.shape)

        if (not tf.get_variable_scope().reuse): 
            # first time, put into weight decay
            wd = tf.multiply( tf.nn.l2_loss(w_var), 
                              self.weight_decay_rate, 
                              name="weight_decay_loss")
            tf.add_to_collection("losses", wd)

        print "-- Initializing VGG conv layer [{}] of shape {}".format(
            w_var.name, w_var.get_shape())

        return w_var

    def get_bias(self, lname, num_classes=None):
        b = self.vgg_params[lname][1]
        num_orig_classes = b.size

        if (num_classes is None) or num_classes==num_orig_classes:
            init = tf.constant_initializer(value=b, dtype=tf.float32)
            bshape_new=[num_orig_classes,]
        else:
            # need to summarise pretrained weights, multiple original
            # classes to one new class, see @_fc_layer
            assert num_classes<num_orig_classes
            num_old_classes_per_new = num_orig_classes//num_classes

            b_new=np.zeros([num_classes,])
            for j in range(0, num_orig_classes, num_old_classes_per_new):
                j_=j+num_old_classes_per_new
                i_new=j//num_old_classes_per_new
                b_new[i_new] = np.mean(b[j:j_])
            init = tf.constant_initializer(value=b_new, dtype=tf.float32)
            bshape_new=[num_classes,]

        b_var = tf.get_variable(name='bias', 
                                initializer=init,
                                shape=bshape_new)
        return b_var

    def get_fc_filter(self, lname, kshape, num_classes=None):
        print "Loading FC-layer {} weights, num_classes={}".format(
                lname, num_classes)
        w=self.vgg_params[lname][0].reshape(kshape)
        if (num_classes is None) or num_classes==kshape[3]:
            init=tf.constant_initializer(value=w,dtype=tf.float32)
            kshape_new=kshape
        else:
            # need to summarise pretrained weights, multiple original
            # classes to one new class, see @_fc_layer
            num_orig_classes = kshape[3] 
            assert num_classes<num_orig_classes
            num_old_classes_per_new = num_orig_classes//num_classes

            w_new=np.zeros(kshape[:3]+[num_classes,])
            for j in range(0, num_orig_classes, num_old_classes_per_new):
                j_=j+num_old_classes_per_new
                i_new=j//num_old_classes_per_new
                w_new[:,:,:,i_new] = np.mean(w[:,:,:,j:j_], axis=3)

                DBG_FLAG=True
                if DBG_FLAG:
                    print "\torig[{}:{}] to new [{}]".format(j,j_,i_new)

            init=tf.constant_initializer(value=w_new,dtype=tf.float32)
            kshape_new=w_new.shape

        w_var=tf.get_variable(name='fc_wt',
                              initializer=init,
                              shape=kshape_new)
        if (not tf.get_variable_scope().reuse):
            wd=tf.multiply(tf.nn.l2_loss(w_var),
                           self.weight_decay_rate,
                           name="weight_decay_loss")
            tf.add_to_collection("losses", wd)
        return w_var
        


