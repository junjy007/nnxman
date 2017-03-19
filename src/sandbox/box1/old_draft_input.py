import os
import itertools
from random import shuffle
import numpy as np
import scipy
import tensorflow as tf


class ImageSegInputData(object):
    def __init__(self, data_dir, opts):
        """
        Representing data for image segmentation.
        Training: image, label is the same size of image, each pixel has a class label.
        Validation
        Test

        NEED:
        - random seed

        - training data file containing the path to each image-label pair
        - test data file
        - valid data file


        """
        self.rand_seed = opts['random_seed']
        self.base_dir = os.path.realpath(os.path.dirname(data_dir))
        self.labelled_data_file = opts['labelled_data_file']

    def get_image_label_loader(self, hypes, training_data_file=None):
        """

        :param hypes:
        :param training_data_file: each line has the form of
            relative_path_image/to/image001.png relative_path_label/to/label_image001.png
          The relative_path is w.r.t. the directiionary of this file.

        :return: a generator, yielding an (image, label) pair per invocation.
          Note when for segmentation task, @label will be an image of per-pixel label.
        """

        rng = np.random.RandomState(self.rand_seed)
        files = [line.rstrip() for line in open(self.labelled_data_file, 'r')]

        for epoche in itertools.count():
            rng.shuffle(files)
            for fpair in files:
                image_file, gt_image_file = fpair.split(" ")
                image_file = os.path.join(self.base_dir, image_file)
                assert os.path.exists(image_file), \
                    "File does not exist: %s" % image_file
                gt_image_file = os.path.join(self.base_dir, gt_image_file)
                assert os.path.exists(gt_image_file), \
                    "File does not exist: %s" % gt_image_file
                image = scipy.misc.imread(image_file, mode='RGB')
                # Please update Scipy, if mode='RGB' is not avaible
                gt_image = scipy.misc.imread(gt_image_file, mode='RGB')
                yield image, gt_image

    def is_shape_fixed(self):
        # TODO
        pass



# Tensorflow-handled data reader

# 1. Get the file names and put them in a queue (optionally, could be shuffled)
# output the queue ( part of the Graph ) -- where to do the enqueue (perhaps by
# coordinator) ?
# 2. From the queue, load an image-label pair (output two place holders)

# The methods here is to be called by the main model building funciton



def read_ (filename_queue):
  """Reads and parses examples from CIFAR10 data files.
  Recommendation: if you want N-way read parallelism, call this function
  N times.  This will give you N independent Readers reading different
  files & positions within those files, which will give better mixing of
  examples.
  Args:
    filename_queue: A queue of strings with the filenames to read from.
  Returns:
    An object representing a single example, with the following fields:
      height: number of rows in the result (32)
      width: number of columns in the result (32)
      depth: number of color channels in the result (3)
      key: a scalar string Tensor describing the filename & record number
        for this example.
      label: an int32 Tensor with the label in the range 0..9.
      uint8image: a [height, width, depth] uint8 Tensor with the image data
  """

  class CIFAR10Record(object):
    pass
  result = CIFAR10Record()

  # Dimensions of the images in the CIFAR-10 dataset.
  # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
  # input format.
  label_bytes = 1  # 2 for CIFAR-100
  result.height = 32
  result.width = 32
  result.depth = 3
  image_bytes = result.height * result.width * result.depth
  # Every record consists of a label followed by the image, with a
  # fixed number of bytes for each.
  record_bytes = label_bytes + image_bytes

  # Read a record, getting filenames from the filename_queue.  No
  # header or footer in the CIFAR-10 format, so we leave header_bytes
  # and footer_bytes at their default of 0.
  reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
  result.key, value = reader.read(filename_queue)

  # Convert from a string to a vector of uint8 that is record_bytes long.
  record_bytes = tf.decode_raw(value, tf.uint8)

  # The first bytes represent the label, which we convert from uint8->int32.
  result.label = tf.cast(
      tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)

  # The remaining bytes after the label represent the image, which we reshape
  # from [depth * height * width] to [depth, height, width].
  depth_major = tf.reshape(
      tf.strided_slice(record_bytes, [label_bytes],
                       [label_bytes + image_bytes]),
      [result.depth, result.height, result.width])
  # Convert from [depth, height, width] to [height, width, depth].
  result.uint8image = tf.transpose(depth_major, [1, 2, 0])

  return result


def inputs(eval_data, data_dir, batch_size):
  """
  Construct input for CIFAR evaluation using the Reader ops.

  :param eval_data:
    eval_data: bool, indicating if one should use the train or eval data set.
    data_dir: Path to the CIFAR-10 data directory.
    batch_size: Number of images per batch.
  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  if not eval_data:
    filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
                 for i in xrange(1, 6)]
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
  else:
    filenames = [os.path.join(data_dir, 'test_batch.bin')]
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

  #? #.examples_per_epoch?
  # now we have filenames

  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer(filenames)

  # Read examples from files in the filename queue.
  read_input = read_cifar10(filename_queue)
  reshaped_image = tf.cast(read_input.uint8image, tf.float32)

  height = IMAGE_SIZE
  width = IMAGE_SIZE

  # Image processing for evaluation.
  # Crop the central [height, width] of the image.
  resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                         width, height)

  # Subtract off the mean and divide by the variance of the pixels.
  float_image = tf.image.per_image_standardization(resized_image)

  # Set the shapes of tensors.
  float_image.set_shape([height, width, 3])
  read_input.label.set_shape([1])

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(num_examples_per_epoch *
                           min_fraction_of_examples_in_queue)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(float_image, read_input.label,
                                         min_queue_examples, batch_size,
                                         shuffle=False)

def TF_AsyncDataFeeder(Object):
    def __init__(self, data_loader_type, data_loader_param,
                 phrase):
        self.data_loader = data_loader_type(data_loader_param)
        dtypes = [tf.float32, tf.float32]

        if self.data_loader.is_shape_fixed(im):
            height = self.data_loader.im_height
            wid

        # build tf-queue
        #def create_queues(hypes, phase):
        #    """Create Queues."""
        #    arch = hypes['arch']
        #    dtypes = [tf.float32, tf.int32]

#            if hypes['jitter']['fix_shape']:
                height = hypes['jitter']['image_height']
                width = hypes['jitter']['image_width']
                channel = hypes['arch']['num_channels']
                num_classes = hypes['arch']['num_classes']
                shapes = [[height, width, channel],
                          [height, width, num_classes]]
            else:
                shapes = None

            capacity = 50
            q = tf.FIFOQueue(capacity=50, dtypes=dtypes, shapes=shapes)
            tf.summary.scalar("queue/%s/fraction_of_%d_full" %
                              (q.name + "_" + phase, capacity),
                              math_ops.cast(q.size(), tf.float32) * (1. / capacity))

            return q


    def build_async_data_feeder(self):
        """
        :return: a data feeder, one can take
        """


    def inputs(hypes, q, phase):
        """

        :param q:
        :param phase:
        :return: (image, label) two operands to be used further to construct the
            computational graph.

        TODO: decouple this step from Tensorflow, allow the image and label to
            be two generic operands
        """

        if phase == 'val':
            image, label = q.dequeue()
            image = tf.expand_dims(image, 0)
            label = tf.expand_dims(label, 0)
            return image, label

        if not hypes['jitter']['fix_shape']:
            image, label = q.dequeue()
            nc = hypes["arch"]["num_classes"]
            label.set_shape([None, None, nc])
            image.set_shape([None, None, 3])
            image = tf.expand_dims(image, 0)
            label = tf.expand_dims(label, 0)
        else:
            image, label = q.dequeue_many(hypes['solver']['batch_size'])

        image = _processe_image(hypes, image)

        # Display the training images in the visualizer.
        tensor_name = image.op.name
        tf.summary.image(tensor_name + '/image', image)

        road = tf.expand_dims(tf.to_float(label[:, :, :, 0]), 3)
        tf.summary.image(tensor_name + '/gt_image', road)

        return image, label
