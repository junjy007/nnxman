import tensorflow as tf
from tensorflow.python.framework import ops, dtypes
import os
import copy
import numpy as np
import logging
import matplotlib.cm as cm


# ==== TENSORFLOW ====

def build_print_shape(pl, msg="", first_n=1):
    return tf.Print(pl, [tf.shape(pl)], message=msg, summarize=5, first_n=first_n)


def build_print_value(pl, msg="", summarise=5, first_n=1):
    return tf.Print(pl, [pl, ], message=msg, summarize=summarise, first_n=first_n)


def test_run(results, steps=1):
    R = []
    with tf.Session() as ss:
        init = tf.global_variables_initializer()
        ss.run(init)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=ss,
                                               coord=coord)

        for i in range(steps):
            R.append(ss.run(results))

        coord.request_stop()
        coord.join(threads)
        ss.close()
    return R


def copy_dict_part(tgt_dict, src_dict, key_list=None):
    """
    Copy some entries from the source list to the target one.
    :param tgt_dict:
    :param src_dict:
    :param key_list: a list has tuple entries as (old_key[, new_key])
    :return:
    """
    key_list = key_list or [(k,) for k in src_dict.keys()]
    for k in key_list:
        if len(k) > 1:
            old_k, new_k = k
        else:
            old_k = k[0]
            new_k = old_k
        tgt_dict[new_k] = copy.deepcopy(src_dict[old_k])


def get_experiment_running_dir(path_setting, exp_id):
    base_dir = path_setting['base']

    d0_ = path_setting['run']
    if not os.path.isabs(d0_):
        d0_ = os.path.join(base_dir, d0_)

    d_ = path_setting['this_experiment']
    if d_ == '#same-as-id':
        d_ = exp_id

    if len(d_) > 0 and d_ != '#none':
        run_dir = os.path.join(d0_, d_)
    else:
        run_dir = d0_

    return run_dir


class CheckpointEvaluationRecord(object):
    """
    Given a checkpoint dir, this object maintains which checkpoints
    have been evaluated and which ones havn't

    """

    def __init__(self, cp_dir, cp_name, eval_rec_file):
        """
        :param cp_dir: checkpoint dir
        :param cp_name: checkpoint filename, the check points are
          cp_dir/cp_name-1.XXX
          cp_dir/cp_name-2.XXX

        :param eval_rec_file: file name of evaluation results, in the directory @cp_dir
          each line of the file is
          /full/path/to/checkpoint/file <losses>
          So the checkpoint path is the string before the first space
        """
        self.eval_rec_path = os.path.join(cp_dir, eval_rec_file)
        self.cp_name = cp_name
        self.cp_dir = cp_dir

    def get_to_eval_checkpoint_paths(self):
        files = os.listdir(self.cp_dir)
        cp_s = list(set([os.path.splitext(f)[0]
                         for f in files
                         if f.startswith(self.cp_name)]))
        maybe_to_eval = [os.path.join(self.cp_dir, f) for f in cp_s]

        eval_done = []
        if os.path.exists(self.eval_rec_path):
            with open(self.eval_rec_path, 'r') as f:
                for l in f:
                    eval_done.append(l.split(' ')[0])

        to_eval = [p for p in maybe_to_eval if p not in eval_done]

        for p in to_eval:
            yield p

    def record_checkpoint_eval_result(self, cp_path, result):
        with open(self.eval_rec_path, 'a') as f:
            f.write("{} {}\n".format(cp_path, result))


# ==== IMAGE ====

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
    :param min_resize_factor_range
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
    stacked_pair = tf.concat((im, lb), axis=ndims - 1)

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


def overlay_image_with_class_colours(im, pb, class_colours=None,
                                     a=0.4):
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
        if np.max(im) <= 1.0:
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
        class_colours = mycm(np.linspace(0, 1.0, num_classes))[:, :3]
    else:
        assert class_colours.shape[0] == num_classes, "#.classess != #.colours"

    over_image = np.zeros_like(im1)
    for c in range(num_classes):
        over_image += np.expand_dims(pb1[:, :, c], axis=2) \
                      * class_colours[c][np.newaxis, np.newaxis, :]

    im1 = im1 * a + over_image * (1 - a)

    return im1.astype(np.uint8)
