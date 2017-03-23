import tensorflow as tf
from experiment_manage.core import Builder, BuilderFactory
# from tensorflow.python.framework import dtypes
# from experiment_manage.util import copy_dict_part, \
#     build_print_value, build_print_shape


class DecoderBuilder(Builder):
    """
    A softmax decoder
    """
    def __init__(self, conf):
        super(DecoderBuilder, self).__init__(conf)
        self.decoded_batch = {}

    def build(self, inputs, phrase):
        """
        :param inputs: list of inputs to decode.
            inputs[0]: the logits produced by the encoder
        :param phrase: train / infer
        :return:
        """
        assert phrase in ['train', 'infer']
        logits = inputs[0]

        if phrase == 'train':
            return [tf.nn.softmax(logits), ]
        else:
            return [tf.nn.softmax(logits), tf.arg_max(logits, dimension=3)]


class DecoderBuilderFactory(BuilderFactory):
    def __init__(self):
        super(DecoderBuilderFactory, self).__init__()

    def get_builder(self, conf):
        return DecoderBuilder(conf)


def id_str():
    return "Architecture construction:", __file__
