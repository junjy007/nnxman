import tensorflow as tf
import numpy as np
import threading
import logging
import sys
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

def build_comp_model(x):
    """
    :param x: a placeholder the input of the model
    :return:
    """
    w = tf.get_variable(name='w',shape=[3,],
                        initializer=tf.constant_initializer(value=[1,2,3],
                                                            dtype=tf.float32))
    y = tf.multiply(x,w)
    return y

def build_feed_queue():
    q = tf.FIFOQueue(capacity=10, dtypes=[tf.float32,], shapes=[[3,],])
    return q

def start_data_feeder(q, ss, data):
    """
    :param q: A TF FIFOQueue object
    :param s: The current running session
    :return:
    """
    # enqueue_op, place holders
    x = tf.placeholder(tf.float32,name="x")
    enqueue_op = q.enqueue((x,))

    def enqueue_loop():
        print "hi data"
        logging.info("enqueue_loop Thread started: ")
        for d in data:
            logging.info("data enqueued: {}".format(d))
            ss.run(enqueue_op, feed_dict={x:d})

    t = threading.Thread(target=enqueue_loop)
    t.start()


if __name__=='__main__':

    q = build_feed_queue()
    x = q.dequeue()[0]
    y = build_comp_model(x)

    ss = tf.Session()
    init=tf.global_variables_initializer()
    ss.run(init)

    data = np.arange(99).reshape((33,3))
    start_data_feeder(q,ss,data)

    print "Data feeder started ..."

    for i in range(99):
        print "data [{}]: out {} ".format(i, ss.run(y))






