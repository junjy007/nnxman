"""
core.py
Model factory produces model.
From Data, Model definition
Perform training
Optionally do evaluation of the check points
"""
import imp
import os
import sys
import logging
import tensorflow as tf
from util import get_experiment_running_dir


class Builder(object):
    def __init__(self, conf, phrase):
        self.conf = conf
        assert phrase == 'train' or phrase == 'valid'
        return

    def build(self, input_list):
        raise Exception("Builder is an interface")


class BuilderFactory(object):
    """
    TODO: Refactor to factories
    """

    def __init__(self):
        return

    def get_builder(self, conf, phrase):
        raise Exception("BuilderFactory is an interface")


# noinspection,SpellCheckingInspection
class ModelFactory(object):
    def __init__(self, conf):
        self.conf = conf

        self.run_dir = get_experiment_running_dir(conf['path'], conf['version']['id'])
        logging.info("Run experiment in {}".format(self.run_dir))

        if conf['version']['running_copy']:  # now all scripts and settings
            # have been copied to the directory exclusively for THIS
            # experiment, ...
            src_base = os.path.join(self.run_dir, 'src')
        else:  # ..., as opposed to running from the project dir, and using
            # the dev-copy of the source code.
            src_base = conf['path']['base']
            logging.warning("Development, using project dir {}".format(src_base))

        for d_ in conf['path']['python_mod']:
            if os.path.isabs(d_):
                sys.path.insert(1, d_)
            else:
                sys.path.insert(1, os.path.join(src_base, d_))

        def load_mod(mod_name):
            mod_file = os.path.join(src_base, conf['model'][mod_name])
            return imp.load_source(mod_name, mod_file)

        self.mods = {_m: load_mod(_m) for _m
                     in ['data_source', 'encoder', 'decoder', 'loss', 'solver']}
        self.training_graph = {}
        self.inference_graph = {}
        self.sess = {}
        self.log_settings = self.get_log_settings()
        self.setup_dirs()
        return

    def train(self):
        with tf.Graph().as_default():
            self.build_training_graph()
            self.start_session('train')

        self.run_training_steps(start_step=self.get_global_step(),
                                end_step=self.conf['solver']['max_steps'])

        self.sess['coord'].request_stop()
        self.sess['coord'].join(self.sess['threads'])

    def evaluate(self):
        print self.mods
        print "- In a new graph"
        print "- Build inference computations"
        print ""
        return

    def build_training_graph(self):
        input_builder_factory = self.mods['data_source'].InputBuilderFactory()
        assert isinstance(input_builder_factory, BuilderFactory)
        input_builder = input_builder_factory.get_builder(self.conf, 'train')
        assert isinstance(input_builder, Builder)

        encoder_builder_factory = self.mods['encoder'].EncoderBuilderFactory()
        assert isinstance(encoder_builder_factory, BuilderFactory)
        encoder_builder = encoder_builder_factory.get_builder(self.conf, 'train')
        assert isinstance(encoder_builder, Builder)

        decoder_builder_factory = self.mods['decoder'].DecoderBuilderFactory()
        assert isinstance(decoder_builder_factory, BuilderFactory)
        decoder_builder = decoder_builder_factory.get_builder(self.conf, 'train')
        assert isinstance(decoder_builder, Builder)

        loss_builder_factory = self.mods['loss'].LossBuilderFactory()
        assert isinstance(loss_builder_factory, BuilderFactory)
        loss_builder = loss_builder_factory.get_builder(self.conf, 'train')
        assert isinstance(loss_builder, Builder)

        solver_builder_factory = self.mods['solver'].TrainBuilderFactory()
        assert isinstance(solver_builder_factory, BuilderFactory)
        solver_builder = solver_builder_factory.get_builder(self.conf, 'train')
        assert isinstance(solver_builder, Builder)

        images, labels = input_builder.build([])
        logits = encoder_builder.build([images, ])[0]
        class_probabilities = decoder_builder.build([logits, ])[0]
        losses = loss_builder.build([class_probabilities, labels])[0]
        train_op = solver_builder.build([losses, ])[0]

        self.training_graph['data_feeder'] = lambda: {}
        self.training_graph['train_op'] = train_op
        return

    def run_training_steps(self, start_step=0, end_step=999):
        """
        """
        sess = self.sess['sess']
        train_op = self.training_graph['train_op']
        data_feeder = self.training_graph['data_feeder']
        saver = self.sess['saver']
        save_steps = self.log_settings['save_every_n_steps']
        summary_steps = self.log_settings['summary_every_n_steps']
        save_path = self.log_settings['save_path']
        global_step = self.sess['global_step']
        global_step_inc = tf.assign_add(global_step, 1)
        summary_op = self.sess['summary_op']
        summary_writer = self.sess['summary_writer']
        assert isinstance(sess, tf.Session)
        assert callable(data_feeder)
        assert isinstance(saver, tf.train.Saver)
        assert isinstance(summary_writer, tf.summary.FileWriter)

        for t in xrange(start_step, end_step):
            sess.run(train_op, feed_dict=data_feeder())
            sess.run(global_step_inc)
            if (t + 1) % save_steps == 0:
                saver.save(sess, save_path=save_path, global_step=global_step)
            if (t + 1) % summary_steps == 0:
                summ_ = sess.run(summary_op)
                summary_writer.add_summary(summ_, global_step=self.get_global_step())
            print "Global step is now {}".format(self.get_global_step())

    def start_session(self, phrase):
        cs = self.log_settings
        global_step = tf.Variable(0, name='global_step', trainable=False)
        summary_op = tf.summary.merge_all()
        saver = tf.train.Saver(
            max_to_keep=cs['max_to_keep'],
            keep_checkpoint_every_n_hours=cs['keep_every_n_hours'])

        sess = tf.Session()
        if cs['load_path']:
            saver.restore(sess, cs['load_path'])
            logging.info("Resuming training from global step {}".format(
                global_step.eval(sess)))
        else:
            sess.run(tf.global_variables_initializer())
            assert phrase == 'train'

        if phrase == 'train':
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            summary_dir = cs['train_summary_dir']
        elif phrase == 'valid':
            coord = None
            threads = None
            summary_dir = cs['valid_summary_dir']
        else:
            raise ValueError("Unexpected phrase {}".format(phrase))
        summary_writer = \
            tf.summary.FileWriter(logdir=summary_dir, graph=sess.graph)
        self.sess['sess'] = sess
        self.sess['coord'] = coord
        self.sess['threads'] = threads
        self.sess['saver'] = saver
        self.sess['global_step'] = global_step
        self.sess['summary_op'] = summary_op
        self.sess['summary_writer'] = summary_writer
        return

    def get_log_settings(self):
        pj = os.path.join
        p = self.conf['path']
        cp_dir = pj(self.run_dir, p['checkpoint_dir'])
        cp_save_path = pj(cp_dir, p['checkpoint_name'])
        cp_state = tf.train.get_checkpoint_state(cp_dir)
        if cp_state and cp_state.model_checkpoint_path:
            cp_load_path = cp_state.model_checkpoint_path
        else:
            cp_load_path = None

        cp = {'save_path': cp_save_path,
              'load_path': cp_load_path}
        for k_ in ['max_to_keep', 'keep_every_n_hours', 'save_every_n_steps',
                   'summary_every_n_steps']:
            cp[k_] = self.conf['checkpoint'][k_]
        for k_ in ['train_summary_dir', 'valid_summary_dir']:
            cp[k_] = pj(self.run_dir, p[k_])
        return cp

    def get_global_step(self):
        assert 'sess' in self.sess and 'global_step' in self.sess
        g_step = self.sess['sess'].run(self.sess['global_step'])
        return g_step

    def setup_dirs(self):
        needed_dirs = {
            'running': self.run_dir,
            'checkpoint': os.path.dirname(self.log_settings['save_path']),
            'train-summary': self.log_settings['train_summary_dir'],
            'valid-summary': self.log_settings['valid_summary_dir']
        }

        for k, v in needed_dirs.iteritems():
            if not os.path.exists(v):
                logging.warning("Creating non-exist {} dir: {} ...".format(k, v))
                os.mkdir(v)
            else:
                logging.info("Using {} dir: {}".format(k, v))
