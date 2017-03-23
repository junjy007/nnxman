import tensorflow as tf
import logging
from tensorflow.python.framework import dtypes
from experiment_manage.util import build_print_value, build_print_shape
from experiment_manage.core import Builder, BuilderFactory


class XentropyLossBuilder(Builder):
    def __init__(self, conf):
        super(XentropyLossBuilder, self).__init__(conf)
        self.class_weights = conf['objective']['class_weights']
        self.debug = conf['debug']

    def build(self, inputs, phrase):
        """
        :param inputs: [pred_batch, label_batch]
        :param phrase: train / infer
        """
        assert phrase == 'train'  # infer shouldn't have access to labels
        pred_batch, label_batch = inputs
        sh = tf.shape(pred_batch)
        q = tf.reshape(pred_batch, (-1, sh[-1]))
        sh_new = tf.shape(q)
        p = tf.reshape(tf.cast(label_batch, dtypes.float32), sh_new)
        elem_loss = -tf.reduce_sum(tf.log(q) * p * self.class_weights, axis=1)
        if self.debug['elem_loss']:
            elem_loss = build_print_shape(elem_loss, "elemement loss:")
        loss = tf.reduce_mean(elem_loss, name='xentropy')
        tf.summary.scalar('xentropy_loss', loss)
        if self.debug['mean_loss']:
            loss = build_print_value(loss, msg="mean loss", first_n=9999)

        # TODO: include penalties such as weight decay
        return [loss, ]


class LossBuilderFactory(BuilderFactory):
    def __init__(self):
        super(LossBuilderFactory, self).__init__()

    def get_builder(self, conf):
        return XentropyLossBuilder(conf)


def id_str():
    return "Architecture construction:", __file__
