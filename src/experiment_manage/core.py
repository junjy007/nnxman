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
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import dtypes
from util import get_experiment_running_dir  #  build_print_shape


DTYPE = dtypes.float32

class Builder(object):
    def __init__(self, conf):
        self.conf = conf
        return

    def build(self, input_list, phrase):
        raise Exception("Builder is an interface")


class BuilderFactory(object):
    """
    TODO: Refactor to factories
    """

    def __init__(self):
        return

    def get_builder(self, conf):
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
        self.validation_graph = {}
        self.sess = {}
        self.log_settings = self.get_log_settings()
        self.setup_dirs()
        return

    def train(self):
        with tf.Graph().as_default():
            self.build_training_graph()
            self.start_training_session()

        self.run_training_steps(start_step=self.get_global_step(),
                                end_step=self.conf['solver']['max_steps'])

        self.sess['coord'].request_stop()
        self.sess['coord'].join(self.sess['threads'])

    def evaluate(self, cp_path):
        with tf.Graph().as_default():
            # Get data feeder for cross-validation
            self.build_validation_graph()
            self.start_inference_session('valid', cp_path)

        losses = self.run_evaluation()
        self.sess['coord'].request_stop()
        self.sess['coord'].join(self.sess['threads'])

        xentropy_losses = [l_[0] for l_ in losses]
        xentropy_losses_mean = np.mean(xentropy_losses)

        return xentropy_losses, xentropy_losses_mean

    def build_training_graph(self):
        input_builder_factory = self.mods['data_source'].LabelledSampleInputBuilderFactory()
        assert isinstance(input_builder_factory, BuilderFactory)
        # noinspection PyArgumentList
        input_builder = input_builder_factory.get_builder(self.conf, 'training')
        assert isinstance(input_builder, Builder)

        encoder_builder_factory = self.mods['encoder'].EncoderBuilderFactory()
        assert isinstance(encoder_builder_factory, BuilderFactory)
        encoder_builder = encoder_builder_factory.get_builder(self.conf)
        assert isinstance(encoder_builder, Builder)

        decoder_builder_factory = self.mods['decoder'].DecoderBuilderFactory()
        assert isinstance(decoder_builder_factory, BuilderFactory)
        decoder_builder = decoder_builder_factory.get_builder(self.conf)
        assert isinstance(decoder_builder, Builder)

        loss_builder_factory = self.mods['loss'].LossBuilderFactory()
        assert isinstance(loss_builder_factory, BuilderFactory)
        loss_builder = loss_builder_factory.get_builder(self.conf)
        assert isinstance(loss_builder, Builder)

        solver_builder_factory = self.mods['solver'].TrainBuilderFactory()
        assert isinstance(solver_builder_factory, BuilderFactory)
        solver_builder = solver_builder_factory.get_builder(self.conf)
        assert isinstance(solver_builder, Builder)

        images, labels = input_builder.build([], 'train')
        logits = encoder_builder.build([images, ], 'train')[0]
        class_probabilities = decoder_builder.build([logits, ], 'train')[0]
        total_loss = loss_builder.build([class_probabilities, labels], 'train')[0]
        train_op = solver_builder.build([total_loss, ], 'train')[0]

        self.training_graph = {'data_feeder': lambda: {}, 'train_op': train_op}
        return self.training_graph

    def build_inference_graph(self, images):
        """
        Build a graph for forward computation only. It is to be used for deployment
        or cross-validation.
        :param images: placeholder of image data. [NxHxWxC]
        :return:
        """
        encoder_builder_factory = self.mods['encoder'].EncoderBuilderFactory()
        assert isinstance(encoder_builder_factory, BuilderFactory)
        encoder_builder = encoder_builder_factory.get_builder(self.conf)
        assert isinstance(encoder_builder, Builder)

        decoder_builder_factory = self.mods['decoder'].DecoderBuilderFactory()
        assert isinstance(decoder_builder_factory, BuilderFactory)
        decoder_builder = decoder_builder_factory.get_builder(self.conf)
        assert isinstance(decoder_builder, Builder)

        logits = encoder_builder.build([images, ], 'infer')[0]
        prob, pred = decoder_builder.build([logits, ], 'infer')

        self.inference_graph = {'prob': prob, 'pred': pred}
        return self.inference_graph

    def build_validation_graph(self):
        """
        Build a graph for cross validation
        :return:
        """
        input_builder_factory = self.mods['data_source'].LabelledSampleInputBuilderFactory()
        assert isinstance(input_builder_factory, BuilderFactory)
        # noinspection PyArgumentList
        input_builder = input_builder_factory.get_builder(self.conf, 'validation')
        assert isinstance(input_builder, Builder)

        loss_builder_factory = self.mods['loss'].LossBuilderFactory()
        assert isinstance(loss_builder_factory, BuilderFactory)
        loss_builder = loss_builder_factory.get_builder(self.conf)
        assert isinstance(loss_builder, Builder)

        # phrase = train : for validation is part of train, with supervision
        images, labels = input_builder.build([], 'train')
        ig = self.build_inference_graph(images)  # ig has ['prob'] and ['pred']

        # dbg = build_print_shape(ig['prob'], "pred-prob")
        # dbg2 = build_print_shape(labels, "labels")

        losses = loss_builder.build([ig['prob'], labels], 'train')

        self.validation_graph = {'data_feeder': lambda: {},
                                 'losses': losses}
        return self.validation_graph

    def run_training_steps(self, start_step=0, end_step=999):
        """
        """
        sess = self.sess['sess']
        train_op = self.training_graph['train_op']
        data_feeder = self.training_graph['data_feeder']
        saver = self.sess['saver']
        save_steps = self.log_settings['save_every_n_steps']
        summary_steps = self.log_settings['summary_every_n_steps']
        save_path = self.log_settings['train_save_path']
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

    def run_evaluation(self):
        # TODO: get evaluation data size and evaluation batch-size
        # TODO: create an op, go through many validation batches and write summary
        sess = self.sess['sess']
        data_feeder = self.validation_graph['data_feeder']
        assert isinstance(sess, tf.Session)
        assert callable(data_feeder)
        num_vsteps = 3  # TODO: use all samples per validation

        losses = []
        for t in range(num_vsteps):
            step_losses = sess.run(self.validation_graph['losses'],
                                   feed_dict=data_feeder())
            logging.debug("val-step-{}, losses: {}".format(t, step_losses))
            losses.append(step_losses)

        return losses

    def start_training_session(self):
        """
        :return:
        """
        cs = self.log_settings
        global_step = tf.Variable(0, name='global_step', trainable=False)
        summary_op = tf.summary.merge_all()
        saver = tf.train.Saver(
            max_to_keep=cs['max_to_keep'],
            keep_checkpoint_every_n_hours=cs['keep_every_n_hours'])
        sess = tf.Session()
        if cs['train_load_path']:
            logging.info("Resume training from {}".format(cs['train_load_path']))
            saver.restore(sess, cs['train_load_path'])
        else:
            logging.info("Start training from scratch")
            sess.run(tf.global_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        summary_writer = tf.summary.FileWriter(logdir=cs['train_summary_dir'],
                                               graph=sess.graph)
        gs = global_step.eval(session=sess)
        logging.info("global_step begin with {}".format(gs))
        self.sess['sess'] = sess
        self.sess['coord'] = coord
        self.sess['threads'] = threads
        self.sess['saver'] = saver
        self.sess['global_step'] = global_step
        self.sess['summary_op'] = summary_op
        self.sess['summary_writer'] = summary_writer
        return

    def start_inference_session(self, phrase, model_checkpoint_path):
        assert phrase in ['valid', 'deploy']
        global_step = tf.Variable(0, name='global_step', trainable=False)
        summary_op = tf.summary.merge_all()
        saver = tf.train.Saver()
        sess = tf.Session()
        saver.restore(sess=sess, save_path=model_checkpoint_path)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)  # if no queue-runners
        # have been constructed, threads will be empty, that's normal
        summary_dir = self.log_settings[phrase + '_summary_dir']
        summary_writer = \
            tf.summary.FileWriter(logdir=summary_dir, graph=sess.graph)
        gs = global_step.eval(session=sess)
        logging.info("Load inference net with saved model {} trained {} steps".format(model_checkpoint_path, gs))
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

        cp = {'train_save_path': cp_save_path,
              'train_load_path': cp_load_path}
        for k_ in ['max_to_keep', 'keep_every_n_hours', 'save_every_n_steps',
                   'summary_every_n_steps']:
            cp[k_] = self.conf['checkpoint'][k_]
        for k_ in ['train', 'valid', 'deploy']:
            dk_ = k_ + '_summary_dir'
            cp[dk_] = pj(self.run_dir, p[dk_])
        return cp

    def get_global_step(self):
        assert 'sess' in self.sess and 'global_step' in self.sess
        g_step = self.sess['sess'].run(self.sess['global_step'])
        return g_step

    def setup_dirs(self):
        needed_dirs = {
            'running': self.run_dir,
            'checkpoint': os.path.dirname(self.log_settings['train_save_path']),
            'train-summary': self.log_settings['train_summary_dir'],
            'valid-summary': self.log_settings['valid_summary_dir'],
            'deploy-summary': self.log_settings['deploy_summary_dir'],
            'tmp': os.path.join(self.run_dir, 'tmp')
        }

        for k, v in needed_dirs.iteritems():
            if not os.path.exists(v):
                logging.warning("Creating non-exist {} dir: {} ...".format(k, v))
                os.mkdir(v)
            else:
                logging.info("Using {} dir: {}".format(k, v))
