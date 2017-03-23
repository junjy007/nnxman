# noinspection SpellCheckingInspection
"""
Experiment building pipeline of input for tensorflow. The functions in
this file would provide two operands, for label and image. This module is meant
to be a dynamic loaded component in model training-evaluation framework.
However, you can also test this script directly as
follows.

Usage:
  input.py <config-file> [--project-dir=<pdir>] [--split=<sp>]

Options
  --project-dir=<pdir>  Overriding the base-path in environmental variable
    $PROJECT_DIR, from which the relative pathes in the experiment configuration
    are constructed.
  --split=<sp>  Which data split to try can be training or validation [default: training]

This module provide a portal function @build_inputs, that returns the input nodes
for a computational graph of image segmentation. See the function documentation.


"""
from docopt import docopt
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import os
import logging
import json
from experiment_manage.util import build_print_shape, build_print_value, test_run
from experiment_manage.core import Builder, BuilderFactory


# import numpy as np
# import itertools
# import scipy.misc


class MyImageProcessor(object):
    """
    Provide @build_image_process and @build_input_pair_process, given input
    image, returns processed image (and label/annotation) nodes.
    """

    def __init__(self, jitter, image_info, class_colours):
        """
        :param jitter, options for making samples from raw image/annotation.
        :param class_colours: colour code of classes
            [[r,g,b]_1, [r,g,b]_2, ... ] class_clr[i] is the colour code for
            i-th class, to be used to interpret the label image
        """
        self.jitter = jitter
        self.raw_im_height = image_info['height']
        self.raw_im_width = image_info['width']
        self.num_channels = image_info['channels']
        self.im_height = self.raw_im_height
        self.im_width = self.raw_im_width

        self.cls_names = []
        self.cls_colours = []
        for k, v in class_colours.iteritems():
            self.cls_names.append(k)
            self.cls_colours.append(v)
        self.num_classes = len(self.cls_names)

    def is_shape_fixed(self):
        flexible = (self.im_height is None) and \
                   (self.im_width is None)
        return not flexible

    def get_shape(self):
        return [self.im_height, self.im_width]

    def build_image_process(self, raw_im, keep_value=False):
        """
        :param raw_im: an image tensor (placeholder), expecting input of
             [height x width x 3-channels], of RGB (default of Tensorflow)
        :param keep_value: to use nearest-neighbour interpolation when scaling,
             so the exact values of pixels are kept. It is useful when dealing
              with annotaiton images.
        """
        # im = tf.image.resize_image_with_crop_or_pad(raw_im,
        #                                            self.trn_im_height,
        #                                            self.trn_im_width)
        # don't worry about the mean, that's model's business
        # TODO: perform processing steps specified in jitter.
        im = raw_im
        assert isinstance(keep_value, bool)
        assert isinstance(self.jitter, dict)
        return im

    def build_input_pair_process(self, raw_im, raw_lb):
        """
        :param raw_im:
        :param raw_lb:
        :   return: processed image and label, TF-placeholders

        Note this function mainly deals with the label image. The input will be
        delegated to @build_input_image_process
        """
        im1 = self.build_image_process(raw_im)
        lb1 = self.build_interpret_label(
            self.build_image_process(raw_lb, keep_value=True))

        im = tf.expand_dims(im1, 0)
        lb = tf.expand_dims(lb1, 0)
        im.set_shape([1, self.raw_im_height, self.raw_im_width, self.num_channels])
        lb.set_shape([1, self.raw_im_height, self.raw_im_width, self.num_classes])
        return im, lb

    def build_interpret_label(self, label_im):
        """
        For each pixel of label image, compare to class-colour code to determine
        class membership.

        :param label_im: image representing class memberships

        :return: [H x W x #.classes]
        """
        label_is_classes = [tf.reduce_all(
            tf.equal(label_im, ops.convert_to_tensor(clr, dtype=dtypes.uint8)),
            reduction_indices=2
        ) for clr in self.cls_colours]

        semantic_label = tf.stack(label_is_classes, axis=2)
        return semantic_label


# noinspection PyShadowingNames
class LabelledSampleInputBuilder(Builder):  # for labelled data samples
    #  """
    #  Portal of building input nodes of a computational graph.
    #
    #  :param opts:
    #  ---- ABOUT DATABASE ORGANISATION ----
    #  - annotated_sample_list, name of a file name, where each line contains a pair
    #      of file names 'path/to/image path/to/annotation_image'. This list is
    #      NOT used when training and validation lists are provided respectively (if
    #      trn_vld_split method is assigned, see below).
    #  - trn_sample_list, see 'annotated_sample_list'
    #  - vld_sample_list, see 'annotated_sample_list'
    #  - trn_vld_split, how to split training, validation and test data
    #    - method: ['random' | 'assigned']
    #    <-- if @method == 'random' -->
    #    - seed: , random seed
    #    - ratio: [trn, vld], integer percentage, must add to 100
    #    <-- if @method == 'assigned' -->
    #    Training and validation samples are provided by the [trn_|vld_]sample_list.
    #  ---- ABOUT DATA-SET ----
    #  - batch_size, size of a mini-batch for training and validation, must be 1
    #    if @shape_fixed is false
    #  - shape_fixed, whether all samples are of the same shape
    #  ---- ABOUT ANNOTATION ----
    #  - class_colours, dictionary {'road',[255,0,255], 'background':[255,0,0]},
    #    representing the RGB colour for each class in the annotation image
    #  ---- ABOUT INPUT IMAGES ----
    #  - jitter, NA
    #  ---- NOT IMPLEMENTED ----
    #  - unannotated_sample_list, name of a file, where each line is an unlabelled
    #      image (for testing), can be used for test or enhance training.
    #  - jitter, data augumentation, how to slightly disturb the data to obtain
    #    more samples.
    #
    #   class_colours: colour-code for different classes
    #   "annotated_sample_list":"DATA/data_road_sml/sample-annotation-list.txt
    #  :return:
    #  """
    DEBUG_INFO = {
        'file_name':False,
        'raw_input': False,
        'input_batch': True
    }

    def __init__(self, conf, data_split):  # noinspection Shadowing
        super(LabelledSampleInputBuilder, self).__init__(conf)
        assert data_split in ['training', 'validation']
        dc = conf['data']
        self.preproc = MyImageProcessor(jitter=dc['jitter'],
                                        image_info=dc['image_info'],
                                        class_colours=dc['class_colours'])
        if os.path.isabs(conf['path']['data']):
            self.data_dir = conf['path']['data']
        else:
            self.data_dir = os.path.join(conf['path']['base'], conf['path']['data'])

        if data_split == 'training':
            lf = dc['trn_sample_list']
            self.batch_size = conf['solver']['batch_size']
        elif data_split == 'validation':
            lf = dc['vld_sample_list']
            self.batch_size = None  # single-sample-batch, this will not be used
        else:
            raise ValueError("Unknown phrase {}".format(data_split))
        self.sample_list_file = os.path.join(self.data_dir, lf)
        self.random_seed = dc['random_seed']
        self.data_split = data_split
        logging.info("Using samples list \n\t{}\n for {}".format(self.sample_list_file, data_split))

    # noinspection PyUnboundLocalVariable
    def build(self, input_list, phrase):
        assert phrase == 'train'  # these are samples with labels both
        # 'training' and 'validation' belonging to train in broader sense
        assert len(input_list) == 0  # Input doesn't have input
        INFO = self.__class__.DEBUG_INFO

        # 1. Get queues of file names
        # TODO: distinguish training, validation and unlabelled
        im_files, lb_files, num_files = self.read_filename_list(self.sample_list_file)
        # vld_im_files, vld_lb_files, num_files = self.read_filename_list(self.vld_sample_list)

        # 2. Load images
        raw_image, raw_label = self.build_pair_queue_reader(im_files, lb_files)
        # We don't jitter validation images, the raw-to-processed just expand the dimension
        # so each sample becomes a single-sample mini-batch

        if INFO['raw_input']:
            raw_image = build_print_shape(raw_image, "Input image ")
            raw_label = build_print_shape(raw_label, "Input label ")

        # 3. Image pre-process before making batches
        # TODO: if training we do jitter for augument, otherwise, skip
        single_sample_batch = True
        if self.data_split == 'training':
            image, label = \
                self.preproc.build_input_pair_process(raw_image, raw_label)
            if self.preproc.is_shape_fixed():
                # if pre-processor returns is_shape_fixed()==True, it is responsible for
                # setting shapes of the trn_image and trn_label
                image_batch, label_batch = \
                    tf.train.batch([image, label],
                                   batch_size=self.batch_size,
                                   enqueue_many=True)
                # enqueue_many: expect small "batches" from image preprocessing, because
                #   data augument may result in multiple samples from one "raw sample"
                single_sample_batch = False
        else:  # no pre-processing for validation and deploy
            image = raw_image
            label = self.preproc.build_interpret_label(raw_label)

        if single_sample_batch:
            image_batch = tf.expand_dims(image, 0)
            label_batch = tf.expand_dims(label, 0)

        assert isinstance(image_batch, tf.Tensor)
        assert isinstance(label_batch, tf.Tensor)
        if INFO['input_batch']:
            image_batch = build_print_shape(
                image_batch, "Image batch [{}]: ".format(self.data_split))
            label_batch = build_print_shape(
                label_batch, "Label batch [{}]: ".format(self.data_split))

        return image_batch, label_batch

    def read_filename_list(self, list_file):
        """
        :param list_file, each line is a pair of string
                   path/to/image01.png path/to/label_image01.png
            Note the filenames are separated by a space.
        :return: the filename, label, converted to tensor
        """
        im_files = []
        lb_files = []

        list_file = os.path.join(self.data_dir, list_file)
        with open(list_file, 'r') as f:
            for pstr in f:
                s0, s1 = pstr.rstrip().split(' ')
                im_files.append(os.path.join(self.data_dir, s0))
                lb_files.append(os.path.join(self.data_dir, s1))
        im_files_t = ops.convert_to_tensor(im_files, dtype=dtypes.string)
        lb_files_t = ops.convert_to_tensor(lb_files, dtype=dtypes.string)
        return im_files_t, lb_files_t, len(im_files)

    def build_pair_queue_reader(self, im_files, lb_files):
        """
        :param im_files: a list of file names of input images
        :param lb_files: a list of file names of annotation (label) images
        :return:
        """
        INFO = self.__class__.DEBUG_INFO
        input_queue = tf.train.slice_input_producer(
            [im_files, lb_files], num_epochs=None, shuffle=True,
            seed=self.random_seed,
            capacity=32
        )
        im_file_name, lb_file_name = input_queue
        if INFO['file_name']:
            im_file_name = build_print_value(
                im_file_name,"input image:", summarise=256, first_n = 999)
            lb_file_name = build_print_value(
                lb_file_name,"label image:", summarise=256, first_n = 999)


        raw_image = tf.image.decode_png(tf.read_file(im_file_name),
                                        channels=3,
                                        dtype=dtypes.uint8)

        raw_label = tf.image.decode_png(tf.read_file(lb_file_name),
                                        channels=3,
                                        dtype=dtypes.uint8)
        return raw_image, raw_label


# noinspection PyShadowingNames
class LabelledSampleInputBuilderFactory(BuilderFactory):
    def __init__(self):
        super(LabelledSampleInputBuilderFactory, self).__init__()

    def get_builder(self, conf, data_split):
        return LabelledSampleInputBuilder(conf, data_split)


if __name__ == '__main__':
    args = docopt(__doc__)

    with open(args['<config-file>'], 'r') as f:
        conf = json.load(f)

    if args['--project-dir']:
        conf['path']['base'] = args['<pdir>']

    input_builder = LabelledSampleInputBuilder(conf, data_split=args['--split'])
    image_batch, label_batch = input_builder.build([], phrase='train')
    test_run([image_batch, label_batch], steps=3)
