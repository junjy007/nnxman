"""
Try image processing steps in TF-computational graph
"""
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import scipy.misc
import os
import numpy as np
TMP_DIR = '../../DATA/tmp'

data_dir='/Users/junli/local/projects/tf-fcn/DATA/data_road_sml'
sample_file='training/gt_image_2/um_road_000000.png'
filename = os.path.join(data_dir,sample_file)

content = tf.read_file(filename)
label_image = tf.image.decode_png(content)
class_clr=[[255,0,255], [255,0,0]]
label_is_classes = [tf.reduce_all(
  tf.equal(label_image, ops.convert_to_tensor(clr,dtype=dtypes.uint8)),
  reduction_indices=2
) for clr in class_clr]

label = tf.stack(label_is_classes, axis=2)



with tf.Session() as ss:
  init = tf.global_variables_initializer()
  ss.run(init)
  out = ss.run(label)
ss.close()

np.save(os.path.join(TMP_DIR,'tmpout.npy'), out)
#scipy.misc.imsave(os.path.join(TMP_DIR,'tmpout.png'), out)
