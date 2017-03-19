import tensorflow as tf
import numpy as np

def prepare_variable():
    with tf.variable_scope("foo"):
        init = tf.constant_initializer(value=np.asarray([1,2,3,4,5,6]),
                dtype=tf.float32)
        v = tf.get_variable("v", initializer=init, shape=[2,3])
        print "Variable '{}' allocated, shape {}".format(
                v.name, v.get_shape())

def reaccess_variable():
    with tf.variable_scope("foo", reuse=True):
        v = tf.get_variable("v")
        print "Variable '{}' has shape {}".format(
                v.name, v.get_shape())

if __name__=='__main__':
    prepare_variable()
    reaccess_variable()


