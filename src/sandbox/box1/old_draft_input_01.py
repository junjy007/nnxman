"""
Experiment building pipeline of input for tensorflow. The functions in
this file would provide two operands, for label and image. This module is for
model training coordinator. However, you can also test this script directly as
follows.

Usage:
  input.py <data-dir> [--labelled-sample-list=<list-filename>]

Options:
  --labelled-sample-list=<list-filename>  List of training examples
    [default: training-samples.txt].
"""
from docopt import docopt
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import os

import numpy as np
import itertools
import scipy.misc

class MyImageProcessor(object):
  def __init__(self, class_colours):
    """

    :param class_colours: colour code of classes
    """
    self.trn_im_width = 300
    self.trn_im_height = 600
    self.num_channels=3
    self.class_clr=class_colours
    self.num_classes = len(class_colours)


  def build_input_image_process(self, raw_im):
    """
    :param raw_im: an image tensor (placeholder), expecting input of
     [height x width x 3-channels], of RGB (default of Tensorflow)
    :return:
    """
    im = tf.image.resize_image_with_crop_or_pad(raw_im,
                                                self.trn_im_height,
                                                self.trn_im_width)

    # TODO: substract mean
    return im

  def build_input_pair_process(self, raw_im, raw_lb,
                               names=['im','lb']):
    """
    :param raw_im:
    :param raw_lb:
    :param class_clr: [[r,g,b]_1, [r,g,b]_2, ... ] class_clr[i] is the colour code for
      i-th class, to be used to interpret the label image
    :return: processed image and label, TF-placeholders

    Note this function mainly deals with the label image. The input will be
    delegated to @build_input_image_process
    """
    im = self.build_input_image_process(raw_im)
    label_im = tf.image.resize_image_with_crop_or_pad(raw_lb,
                                                self.trn_im_height,
                                                self.trn_im_width)
    lb = self._build_interpret_label(label_im)
    im.set_shape([self.trn_im_height, self.trn_im_width, self.num_channels])
    lb.set_shape([self.trn_im_height, self.trn_im_width, self.num_classes])
    # TODO: consider pixels belonging to no class (without matching colour)
    return im, lb

  def _build_interpret_label(self, label_im):
    """
    For each pixel of label image, compare to class-colour code to determine
      class membership.

    :param label_im: image representing class memberships

    :return: [H x W x #.classes]
    """
    label_is_classes = [tf.reduce_all(
      tf.equal(label_im, ops.convert_to_tensor(clr, dtype=dtypes.uint8)),
      reduction_indices=2
    ) for clr in self.class_clr]

    semantic_label = tf.stack(label_is_classes, axis=2)
    return semantic_label

class MyData(object):
  BUILD_DEBUG_INFO = {
    'raw_input': False,
    'input_batch': True
  }

  def __init__(self, data_dir,
               labelled_listfile,
               preprocessor):
    """
    :param data_dir:
    :param labelled_listfile:
    :param preprocessor: helper object for building TF graph.
     image, label = .build_input_pair_process(raw_image, raw_label)

    """
    self.data_dir = os.path.realpath(data_dir)
    self.batch_size=32
    self.raw_trn_im_width = 1242
    self.raw_trn_im_height = 375
    self.num_channels = 3
    self.preproc = preprocessor
    self.rand_seed=0
    self.labelled_listfile=labelled_listfile



  def read_filename_list(self):
    """
    :param list_file: each line is a pair of string
     path/to/image01.png path/to/label_image01.png
    Note the filenames are separated by a space.
    :return: the filename and label
    """
    im_files = []
    lb_files = []

    list_file = os.path.join(self.data_dir, self.labelled_listfile)
    with open(list_file,'r') as f:
      for pstr in f:
        s0, s1 = pstr.rstrip().split(' ')
        im_files.append(os.path.join(self.data_dir, s0))
        lb_files.append(os.path.join(self.data_dir, s1))
    return im_files, lb_files

  def build_input_pipeline(self):
    INFO = self.__class__.BUILD_DEBUG_INFO

    im_files, lb_files = self.read_filename_list()
    im_files_t = ops.convert_to_tensor(im_files, dtype=dtypes.string)
    lb_files_t = ops.convert_to_tensor(lb_files, dtype=dtypes.string)
    self.train_input_queue = tf.train.slice_input_producer(
      [im_files_t, lb_files_t], num_epochs=None, shuffle=True, seed=42,
      capacity=self.batch_size*2
    )
    file_content0 = tf.read_file(self.train_input_queue[0])
    self.raw_input_image = tf.image.decode_png(file_content0,
                                               channels=3,
                                               dtype=dtypes.uint8,
                                               name='raw_input_image')
    # TODO 1: making multi-batches only when training images have same shape
    # TODO 2: split the labelled samples into training and validation sets

    file_content1 = tf.read_file(self.train_input_queue[1])
    self.raw_label_image = tf.image.decode_png(file_content1,
                                               channels=3,
                                               dtype=dtypes.uint8,
                                               name='raw_label_image')
    if INFO['raw_input']:
      self.raw_input_image=tf.Print(self.raw_input_image,
                                    [tf.shape(self.raw_input_image)],
                                    message="Raw input has shape: ",
                                    summarize=4,
                                    first_n=100)
      self.raw_label_image = tf.Print(self.raw_label_image,
                                      [tf.shape(self.raw_label_image)],
                                      message="Raw label has shape: ",
                                      summarize=4,
                                      first_n=100)

    # Image process before making batches
    self.image, self.label = \
      self.preproc.build_input_pair_process(self.raw_input_image,
                                            self.raw_label_image,
                                            names=['image',
                                                   'label'])

    self.image_batch, self.label_batch = \
      tf.train.batch([self.image, self.label],
                     batch_size=self.batch_size)


    if INFO['input_batch']:
      self.image_batch = \
        tf.Print(self.image_batch,
                 [tf.shape(self.image_batch)],
                  message="Image batch has shape: ",
                  summarize=4,
                  first_n=100)

    self.label_batch = \
      tf.Print(self.label_batch,
               [tf.shape(self.label_batch)],
               message="Label batch has shape: ",
               summarize=4,
               first_n=100)

def test_read(oprand1, oprand2):

  with tf.Session() as ss:
    init = tf.global_variables_initializer()
    ss.run(init)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=ss,
                                           coord=coord)

    for i in range(10):
      ss.run(oprand1)
      ss.run(oprand2)

    coord.request_stop()
    coord.join(threads)
    ss.close()


def build_inputs(opts):
  prep = MyImageProcessor(class_colours=opts['class_colours'])
  indata = MyData(opts['data_dir'],
                  opts['training_sample_list'],
                  prep)
  return indata.image_batch, indata.label_batch

if __name__=='__main__':
  args = docopt(__doc__)
  prep = MyImageProcessor(class_colours=[[255, 0, 255], [255, 0, 0]])
  indata = MyData(args['<data-dir>'],
                  args['--labelled-sample-list'],
                  prep)


  indata.build_input_pipeline()
  test_read(indata.image_batch, indata.label_batch)
