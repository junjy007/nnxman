import imp
import json
import os
import tensorflow as tf
import sys

sys.path.insert(1, "../model")
from myutil import test_run # noinspection PyPep8


my_dir = os.path.realpath(os.path.dirname(__file__))   # $PROJECT_DIR/src/sandbox
project_dir = os.path.split(os.path.split(my_dir)[0])[0]  # ../..
conf_file = os.path.join(project_dir, 'src/config/road_small_test.json')
with open(conf_file, 'r') as f:
    conf = json.load(f)


def get_mod_file(mod_name):
    return os.path.join(project_dir, 'src/model/{}.py'.format(mod_name))


input_mod = imp.load_source('input_mod', get_mod_file('input'))
arch_mod = imp.load_source('arch_mod', get_mod_file('architecture'))
obj_mod = imp.load_source('obj_mod', get_mod_file('loss_kitti_seg'))
trn_mod = imp.load_source('trn_mod', get_mod_file('solver'))

trn_image_batch, trn_label_batch, \
    vld_image, vld_label = input_mod.build_inputs(conf, project_dir)

num_classes = len(conf['data']['class_colours'])

trn_logit_batch = arch_mod.build_encoder(trn_image_batch, num_classes,
                                         'train', conf, project_dir)

trn_predict_batch = obj_mod.build_decoder(trn_logit_batch, conf)

trn_loss = obj_mod.build_loss(trn_predict_batch, trn_label_batch, conf)

trn_op = trn_mod.build_train(trn_loss, conf)

########

class SessionManager(object):
    """
    Handling sessions
    """
    def __init__(self, opts):
        self.sess = None

        # options
        self.saver_opts = {
            'max_to_keep': opts.get('max_to_keep', 1000),
            'keep_checkpoint_every_n_hours': opts.get('keep_checkpoint_every_n_hours', 24)
        }
        self.saver = tf.train.Saver(**self.saver_opts)

        # start
        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)

        self.coord = tf.train.Coordinator
        self.threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)

        # TODO: make summary ops and writers to visualise the training process



    def close(self):
        if isinstance(self.sess, tf.Session):
            self.sess.close()
        pass

########
rec = test_run([trn_op, ], steps=10)

print_value = True
for i, r in enumerate(rec):
    for j, s in enumerate(r):
        if print_value:
            print "Step {}, result {}, shape {}, val {}".format(i, j, s.shape, s)
        else:
            print "Step {}, result {}, shape {}".format(i, j, s.shape)


print "Finished"
