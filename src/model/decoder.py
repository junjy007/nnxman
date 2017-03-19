import tensorflow as tf
from experiment_manage.core import Builder, BuilderFactory
# from tensorflow.python.framework import dtypes
# from experiment_manage.util import copy_dict_part, \
#     build_print_value, build_print_shape


class DecoderBuilder(Builder):
    """
    A softmax decoder
    """
    def __init__(self, conf, phrase):
        super(DecoderBuilder, self).__init__(conf, phrase)
        self.decoded_batch = {}
        self.phrase = phrase

    def build(self, inputs):
        """
        :param inputs: list of inputs to decode.
            inputs[0]: the logits produced by the encoder
        :return:
        """
        logits = inputs[0]
        return [tf.nn.softmax(logits), ]


class DecoderBuilderFactory(BuilderFactory):
    def __init__(self):
        super(DecoderBuilderFactory, self).__init__()

    def get_builder(self, conf, phrase):
        return DecoderBuilder(conf, phrase)


def id_str():
    return "Architecture construction:", __file__
