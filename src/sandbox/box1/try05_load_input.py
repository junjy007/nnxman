"""
Experiment building pipeline of input for tensorflow. The functions in
this file would provide two operands, for label and image.

Usage:
  try05_load_input.py <data-dir> [--labelled-sample-list=<list-filename>]

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





#### BUILDING QUEUE MANUUALLY ####
  # def generate_raw_data_file(self, phrase):
  #   rng = np.random.RandomState(self.rand_seed)
  #   files = [line.rstrip() for line in open(self.labelled_listfile,'r')]
  #
  #   for epoche in itertools.count():
  #     rng.shuffle(files)
  #     for fpair in files:
  #       image_file, gt_image_file = fpair.split(" ")
  #       image_file = os.path.join(self.base_dir, image_file)
  #       assert os.path.exists(image_file), \
  #         "File does not exist: %s" % image_file
  #       gt_image_file = os.path.join(self.base_dir, gt_image_file)
  #       assert os.path.exists(gt_image_file), \
  #         "File does not exist: %s" % gt_image_file
  #       image = scipy.misc.imread(image_file, mode='RGB')
  #       # Please update Scipy, if mode='RGB' is not avaible
  #       gt_image = scipy.misc.imread(gt_image_file, mode='RGB')
  #       yield image, gt_image
  #
  # def generate_labelled_samples(self, phrase):
  #   for image, gt_image in self.generate_raw_data_file(phrase):
  #     gt_bg = np.all(gt_image == self.background_color, axis=2)
  #     gt_road = np.all(gt_image == self.road_color, axis=2)
  #     imh,imw = gt_bg.shape
  #     gt_bg = gt_bg.reshape(imh,imw,1)
  #     gt_road = gt_road.reshape(imh,imw,1)
  #     gt_image = np.concatenate((gt_bg, gt_road), axis=2)
  #
  #     yield image, gt_image
  #     #if phase == 'val':
  #     #  yield image, gt_image
  #     #elif phase == 'train':
  #     #  yield jitter_input(hypes, image, gt_image)
  #     #  yield jitter_input(hypes, np.fliplr(image), np.fliplr(gt_image))
  #
  # def create_queue(self):
  #   dtypes = dtypes = [tf.float32, tf.int32]
  #   shapes = [[self.im_height, self.im_width, 3],
  #             [self.im_height, self.im_width, self.num_classes]]
  #   self.q = tf.FIFOQueue(capacity=50, dtypes=dtypes, shapes=shapes)
  #
  # def get_input_operands(self):
  #   image, label = self.q.dequeue_many(self.batch_size)
  #   return image, label

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

  pass



if __name__=='__main__':
  args = docopt(__doc__)
  prep = MyImageProcessor(class_colours=[[255, 0, 255], [255, 0, 0]])
  indata = MyData(args['<data-dir>'],
                  args['--labelled-sample-list'],
                  prep)


  indata.build_input_pipeline()
  test_read(indata.image_batch, indata.label_batch)




# def read_images_from_disk(input_queue):
#   """Consumes a single filename and label as a ' '-delimited string.
#   Args:
#     filename_and_label_tensor: A scalar string tensor.
#   Returns:
#     Two tensors: the decoded image, and the string label.
#   """
#   label = input_queue[1]
#   file_contents = tf.read_file(input_queue[0])
#   example = tf.image.decode_png(file_contents, channels=3)
#   return example, label
#
#
# # Reads pfathes of images together with their labels
# image_list, label_list = read_labeled_image_list(filename)
#
# images = ops.convert_to_tensor(image_list, dtype=dtypes.string)
# labels = ops.convert_to_tensor(label_list, dtype=dtypes.int32)
#
# # Makes an input queue
# input_queue = tf.train.slice_input_producer([images, labels],
#                                             num_epochs=num_epochs,
#                                             shuffle=True)
#
# image, label = read_images_from_disk(input_queue)
#
# # Optional Preprocessing or Data Augmentation
# # tf.image implements most of the standard image augmentation
# image = preprocess_image(image)
# label = preprocess_label(label)
#
# # Optional Image and Label Batching
# image_batch, label_batch = tf.train.batch([image, label],
#                                           batch_size=batch_size)