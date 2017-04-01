# noinspection SpellCheckingInspection
"""
Experiment building pipeline of input for tensorflow. The functions in
this file would provide two operands, for label and image. This module is meant
to be a dynamic loaded component in model training-evaluation framework.
However, you can also test this script directly as
follows.

Usage:
  input.py <config-file> [--project-dir=<pdir>] [--split=<sp>] [--out-dir=<od>]

Options
  --project-dir=<pdir>  Overriding the base-path in environmental variable
    $PROJECT_DIR, from which the relative pathes in the experiment configuration
    are constructed.
  --split=<sp>  Which data split to try can be training or validation [default: training]
  --out-dir=<sp>  A directory to save the extracted sample images for inspection

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


# TODO: move the image data augment functions to a utility package
# noinspection PyUnresolvedReferences
def _compute_random_resize_factor_range(src_im_shape,
                                        tgt_im_shape,
                                        resize_factor_range,
                                        min_resize_factor_range=0.05):
    # type: (tf.Tensor, tf.Tensor, tf.Tensor, float) -> tuple
    """
    When
      i) the image will be randomly resized, and
      ii) the size of the output image is assigned,
    we need to guarantee
      i) the resized image is larger than the required output size,
         so an output image can be cropped from the resized image.
      ii) the max resize factor >= the min resize factor + min_resize_range


    E.g. if the input image is of [300 x 500], the final output should be
    [20 x 50], the initial resize factor range is given as [0.09, 1.5], then
    the lower range 0.09 is illegal (500 x 0.09 = 45, and 45 < 50), and will
    be enlarged to 0.1.

    NOTE: A very small risk is that numerical inaccuracy can cause problem
    here: 0.1x50 can result in 49.99999, after casting into integer, it
    becomes 49.

    :param src_im_shape:
    :param tgt_im_shape:
    :param resize_factor_range:
    :param min_resize_range
    :return:
    """

    resize_min0 = resize_factor_range[0]
    resize_max0 = resize_factor_range[1]

    # minimum resize factor, so the resized image is larger than / equal to the
    # intended processed image in all dimensions.
    r_min_h = tf.maximum(tf.divide(tgt_im_shape[0], src_im_shape[0]), resize_min0)
    r_min_w = tf.maximum(tf.divide(tgt_im_shape[1], src_im_shape[1]), resize_min0)
    r_min = tf.maximum(r_min_h, r_min_w)

    # ensure minimum resize factor < maximum resize factor
    r_max = tf.maximum(r_min + min_resize_factor_range, resize_max0)
    return r_min, r_max


def random_resize_image_label(im, lb,
                              resize_factor_min,
                              resize_factor_max,
                              min_height=None,
                              min_width=None):
    """
    Resize an (image, label) pair by a random factor between (resize_factor_min,
    resize_factor_max). The resizing will be checked by min_height/width, if provided.

    See also @resize_image_label_with_min_check.

    TODO
    - accept random seed to produce repeatable results.
    """
    resize_factor = tf.random_uniform(shape=(1,),
                                      minval=resize_factor_min,
                                      maxval=resize_factor_max,
                                      dtype=dtypes.float32)[0]
    return resize_image_label_with_min_check(im, lb, resize_factor,
                                             min_height, min_width)


def resize_image_label_with_min_check(im, lb, resize_factor,
                                      min_height=None, min_width=None):
    """
    Resize an (image, label) pair. The label is of same kind of object as the image,
    i.e. a tensor of 3D or 4D [HxWxC] or [NxHxWxC], where each pixel in the image has a
    corresponding label. This is usually used for data augmentation in image
    (semantic) segmentation experiments.

    The resizing will be checked by min_height/width, if provided, to guarantee the
    resized image has a minimum size -- often that is for a following (random) cropping.

    DESIGN
    Q: Why random resizing for a pair, instead of calling a function random resizing
       single image?
    A: i) The factor of resizing for the image and label-image must be the same, so
          "to randomise the resize factor" happens only once.
       ii) resizing image is done by bilinear interpolation; resizing label-image
           is done by nearest neighbouring -- "pixels" in the label-image are class
           memberships, it makes no sense to perform arithmetic operations on them.

    """
    in_h = tf.cast(tf.shape(im)[-3], dtypes.float32)  # suits both batch NxHxWxC
    in_w = tf.cast(tf.shape(im)[-2], dtypes.float32)  # and single image HxWxC
    if min_width:
        min_width = tf.cast(min_width + 1, dtypes.float32)
        # +1 for avoiding numerical issues
        min_width_checked_factor = min_width / in_w
        resize_factor = tf.maximum(resize_factor, min_width_checked_factor)
    if min_height:
        min_height = tf.cast(min_height + 1, dtypes.float32)
        min_height_checked_factor = min_height / in_h
        resize_factor = tf.maximum(resize_factor, min_height_checked_factor)

    resize_h = tf.cast(resize_factor * in_h, dtypes.int32)
    resize_w = tf.cast(resize_factor * in_w, dtypes.int32)

    resized_im = tf.cast(tf.image.resize_images(im, (resize_h, resize_w)), dtypes.uint8)
    resized_lb = tf.image.resize_images(lb, (resize_h, resize_w),
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return resized_im, resized_lb


def random_crop_image_label(im, lb, out_height, out_width):
    """
    Randomly crop an area from image and label-image pair. The cropped area is specified by
    out_height and out_width. The offset is draw randomly. The offset is the same in both
    the image and label-image -- see also @random_resize_image_label for reasons why
    random processing of image and label-image are done in one function, rather than
    calling function processing a single image twice.

    :param im:
    :param lb:
    :param out_height:
    :param out_width:
    :return:
    """
    # stack image and label along the last axis to perform an random
    # but consistent crop
    im_shape = tf.shape(im)
    im_channels = im_shape[-1]
    lb_shape = tf.shape(lb)
    lb_channels = lb_shape[-1]
    ndims = im_shape.get_shape()[0]  # batch or single should be determined statically
    stacked_pair = tf.concat((im, lb), axis=ndims-1)

    if ndims == 3:
        crop_size = ops.convert_to_tensor([out_height, out_width, im_channels + lb_channels])
        stacked_pair_crop = tf.random_crop(stacked_pair, size=crop_size)
        cropped_im = stacked_pair_crop[:, :, :im_channels]
        cropped_lb = stacked_pair_crop[:, :, im_channels:]
    elif ndims == 4:
        crop_size = ops.convert_to_tensor(
            [im_shape[0], out_height, out_width, im_channels + lb_channels])
        stacked_pair_crop = tf.random_crop(stacked_pair, size=crop_size)
        cropped_im = stacked_pair_crop[:, :, :, :im_channels]
        cropped_lb = stacked_pair_crop[:, :, :, im_channels:]
    else:
        logging.error("image shape must be 3d or 4d, now having {}".format(ndims))
        raise ValueError("image shape")
    return cropped_im, cropped_lb


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
        This is the main interface, generating pre-processed image from raw
        image and label image.

        :param raw_im:
        :param raw_lb: a SINGLE image and corresponding label image.
        :   return: processed image and label, TF-placeholders


        This interface should NOT accept batch image. Consider the case of
        data augmentation by random jittering, e.g. random resizing followed by
        random crop. You will want each image is jitterred individually.

        Note this function mainly deals with the label image. The input will be
        delegated to @build_input_image_process
        """
        raw_im_shape = tf.shape(raw_im)
        ndims = raw_im_shape.get_shape()[0]  # should be able to determine *statically*
        assert ndims == 3, "Raw image must be a 3D tensor"
        imh = self.jitter['trn_im_height']
        imw = self.jitter['trn_im_width']
        if self.jitter['random_resize']:
            im_1, lb_1 = random_resize_image_label(
                raw_im, raw_lb,
                resize_factor_min=self.jitter['random_resize_factor_min'],
                resize_factor_max=self.jitter['random_resize_factor_max'],
                min_height=imh,
                min_width=imw)
        else:
            im_1, lb_1 = resize_image_label_with_min_check(raw_im, raw_lb, 1.0, imh, imw)
            # to guarantee the image is enough to crop

        if self.jitter['random_crop']:
            im_2, lb_2 = random_crop_image_label(im_1, lb_1, imh, imw)
        else:
            # crop the image and label image from the centre to ensure
            # the result of pre-processing are of the same size
            im_2 = tf.image.resize_image_with_crop_or_pad(im_1, imh, imw)
            lb_2 = tf.image.resize_image_with_crop_or_pad(lb_1, imh, imw)

        lb_3 = self.build_interpret_label(lb_2)

        im = tf.expand_dims(im_2, 0)
        lb = tf.expand_dims(lb_3, 0)
        im.set_shape([1, imh, imw, self.num_channels])  # both facilitate assertion
        lb.set_shape([1, imh, imw, self.num_classes])
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
        'file_name': False,
        'raw_input': False,
        'input_batch': False
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
        self.graph = {}

    def get_class_colours(self):
        return np.asarray(self.preproc.cls_colours)
        # this colour code [0,:] <==> 1st class, [1,:] <==> 2nd class, ...

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

        self.graph['im_files'] = im_files
        self.graph['lb_files'] = lb_files
        self.graph['raw_image'] = raw_image
        self.graph['raw_label'] = raw_label
        self.graph['image_batch'] = image_batch
        self.graph['label_batch'] = label_batch
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
                im_file_name, "input image:", summarise=256, first_n=999)
            lb_file_name = build_print_value(
                lb_file_name, "label image:", summarise=256, first_n=999)

        raw_image = tf.image.decode_png(tf.read_file(im_file_name),
                                        channels=3,
                                        dtype=dtypes.uint8)

        raw_label = tf.image.decode_png(tf.read_file(lb_file_name),
                                        channels=3,
                                        dtype=dtypes.uint8)

        self.graph['im_file_name_from_queue'] = im_file_name
        self.graph['lb_file_name_from_queue'] = lb_file_name
        return raw_image, raw_label


# noinspection PyShadowingNames
class LabelledSampleInputBuilderFactory(BuilderFactory):
    def __init__(self):
        super(LabelledSampleInputBuilderFactory, self).__init__()

    # noinspection PyMethodOverriding
    def get_builder(self, conf, data_split):
        return LabelledSampleInputBuilder(conf, data_split)


import numpy as np
import matplotlib.cm as cm
def overlay_image_with_class_colours(im, pb, class_colours=None,
                                     a = 0.4):
    # type: (np.ndarray, np.ndarray, np.ndarray, float) -> np.ndarray
    """
    Overlay colours to an image to indiate the predicted (or ground truth)
    likelihood of each pixel belonging to different classes.

    :param im: the input image, RGB, [H x W x 3]
    :param pb: the likelihood of each pixel belonging to the classes
     [H x W x C], C is the number of classes, pb[i,j,:].sum() == 1.0
    :param class_colours: [ C x 3 ], colour for the classes, can be None, then
      the colours will be automatiically allocated
    :param a: a*im + (1-a)*class-colours

    :return:
    """

    num_classes = pb.shape[2]

    if 'float' in im.dtype.name:
        if np.max(im)<=1.0:
            im1 = im * 255.0
        else:
            im1 = im.copy()
    elif 'int' in im.dtype.name:
        im1 = im.astype(np.float32)
    else:
        raise TypeError(
            "Image has type {}, where float or int is "
            "needed".format(im.dtype.name))


    if 'float' in pb.dtype.name:
        pb1 = pb
    elif 'bool' in pb.dtype.name:
        pb1 = pb.astype(np.float32)
    else:
        raise TypeError(
            "Probability has type {}, where float or bool is "
            "needed".format(pb.dtype.name))

    if class_colours is None:
        mycm = cm.get_cmap('bwr')
        class_colours = mycm( np.linspace(0, 1.0, num_classes) )[:,:3]
    else:
        assert class_colours.shape[0] == num_classes, "#.classess != #.colours"

    over_image = np.zeros_like(im1)
    for c in range(num_classes):
        over_image += np.expand_dims(pb1[:, :, c], axis=2) \
                      * class_colours[c][np.newaxis, np.newaxis, :]

    im1 = im1 * a + over_image * (1-a)

    return im1.astype(np.uint8)

def test_run_save_image(image_batch, label_batch, steps,
                        class_colours=None,
                        image_save_dir='',
                        image_save_prefix='trn'):
    from scipy.misc import imsave
    for s in range(steps):
        ims, lbs = test_run([image_batch, label_batch], steps=1)[0]
        # [0] to extract the results of the first (only) running step
        # The images and labels are in batch-form, i.e. multiple images
        assert ims.shape.shape == 4 and ims.shape[0] == 1, \
            "Image batch containing a single image is expected, while " \
            "got shape of {}".format(ims.shape)

        out_im = overlay_image_with_class_colours(\
            ims[0], lbs[0], class_colours=class_colours)

        im_fname = os.path.join(
            image_save_dir,
            image_save_prefix+'_im_{:06d}.png'.format(s))
        lb_fname = os.path.join(
            image_save_dir,
            image_save_prefix+'_lb_{:06d}.png'.format(s))

        imsave(im_fname, out_im)

if __name__ == '__main__':
    args = docopt(__doc__)

    with open(args['<config-file>'], 'r') as f:
        conf = json.load(f)

    if args['--project-dir']:
        conf['path']['base'] = args['<pdir>']

    input_builder = LabelledSampleInputBuilder(conf, data_split=args['--split'])
    image_batch, label_batch = input_builder.build([], phrase='train')
    test_run([image_batch, label_batch], steps=3)
