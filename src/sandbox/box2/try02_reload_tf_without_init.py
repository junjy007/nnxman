"""
Usage:
    try.py (init|reload) --cp-dir=<dir>

"""
import tensorflow as tf
import docopt

if __name__ == '__main__':
    args = docopt.docopt(__doc__)

    if args['init']:
        w = tf.get_variable(name='w', initializer=[[1., 2., 3.], [3, 4, 5]],
                            dtype=tf.float32, trainable=True)
        saver = tf.train.Saver()
        ss = tf.Session()
        ss.run(tf.global_variables_initializer())
        saver.save(sess=ss,save_path=args['--cp-dir'])
        print "Init variable w:"
        print ss.run(w)
    elif args['reload']:
        w = tf.get_variable(name='w', shape=[2,3], dtype=tf.float32,
                            trainable=False)
        saver = tf.train.Saver()
        ss = tf.Session()
        ss.run(tf.global_variables_initializer())
        # Note, no init for w has been done at all!
        saver.restore(sess=ss,save_path=args['--cp-dir'])
        print "Restored variable w:"
        print ss.run(w)




