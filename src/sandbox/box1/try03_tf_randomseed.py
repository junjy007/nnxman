import tensorflow as tf

INIT1 = False
if INIT1:
    g=tf.Graph()
    with g.as_default():
        tf.set_random_seed(1)
        with tf.variable_scope("s0"):
            init=tf.truncated_normal_initializer(stddev=0.001,dtype=tf.float32)
            w1=tf.get_variable("w1", initializer=init, shape=[5,])
            w1=tf.Print(w1,[w1], message="w1: ", summarize=4, first_n=1)
            w2 = tf.get_variable("w2", initializer=init, shape=[5, ])
            w2 = tf.Print(w2, [w2], message="w2: ", summarize=4, first_n=1)

        ss=tf.Session()
        do_init=tf.global_variables_initializer()
        ss.run(do_init)
        wv1 = ss.run(w1)
        wv2 = ss.run(w2)
else:
    g = tf.Graph()
    with g.as_default():
        tf.set_random_seed(1)
        with tf.variable_scope("s0"):
            init = tf.truncated_normal_initializer(stddev=0.001, dtype=tf.float32)
            w1 = tf.get_variable("w1", initializer=init, shape=[10, ])
            w1 = tf.Print(w1, [w1], message="w1: ", summarize=4, first_n=1)


        ss = tf.Session()
        do_init = tf.global_variables_initializer()
        ss.run(do_init)

        wv = ss.run(w1)
        wv1 = wv[:5]
        wv2 = wv[5:]

print "Initialised as", wv1
print wv2




