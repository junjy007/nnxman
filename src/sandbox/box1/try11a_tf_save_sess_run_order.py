import tensorflow as tf
import os
import time

project_dir = os.environ['PROJECT_DIR']
save_dir = os.path.join(project_dir, 'RUNS')
checkpoint_dir = os.path.join(save_dir, 'tmp-a')
checkpoint_path = os.path.join(checkpoint_dir, 'mychkpt')

def build_graph():
    w = tf.Variable([0, 2])
    c = tf.constant([1, 2])
    s = tf.assign_add(w, c)
    return s, w

s, w = build_graph()
global_step = tf.Variable(0, name='global_step', trainable=False)

t0 = time.time()
with tf.Session() as ss:
    ss.run(tf.global_variables_initializer())
    # Create saver AFTER session has begun running.
    # It doesn't matter
    saver = tf.train.Saver()
    cp_state = tf.train.get_checkpoint_state(checkpoint_dir)
    if cp_state and cp_state.model_checkpoint_path:
        print cp_state.model_checkpoint_path
        saver.restore(ss, cp_state.model_checkpoint_path)
    start = global_step.eval()
    for i in range(start, 5000):
        ss.run(s)
        global_step.assign(i).eval()
        print "After running step-{} w={}".format(i, w.eval(ss))
        if (i+1) % 10 == 0:
            saver.save(ss, checkpoint_path, global_step=global_step)
        time.sleep(0.1)

        t1 = time.time()
        if t1 - t0 > 2.0:  # only allow to run 2.0 seconds
            break
