"""
Test if data feeder works fine
"""

import os
import unittest
import json
import sys
sys.path.insert(0, '..')
import model.input as inp
import tensorflow as tf
from tensorflow.python.framework import dtypes, ops
import numpy as np
import logging
from scipy.misc import imsave

logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)s %(funcName)s: %(message)s')


TRIVIAL_TESTS = False

class InputTests(unittest.TestCase):
    def setUp(self):
        with open('test001.json', 'r') as f:
            self.conf = json.load(f)
        self.input_builder = inp.LabelledSampleInputBuilder(self.conf, 'training')
        image_batch, label_batch = self.input_builder.build([], 'train')
        assert isinstance(image_batch, tf.Tensor)
        assert isinstance(label_batch, tf.Tensor)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.coord = tf.train.Coordinator()
        self.feed_threads = tf.train.start_queue_runners(self.sess, self.coord)


    def tearDown(self):
        self.coord.request_stop()
        self.coord.join(self.feed_threads)
        self.sess.close()
        tf.reset_default_graph()


    def test_resize_image_label_with_min_check(self):
        image_v = np.zeros((5,10,3), dtype=np.uint8)
        label_v = np.zeros((5,10,3), dtype=np.uint8)
        image = ops.convert_to_tensor(image_v)
        label = ops.convert_to_tensor(label_v)
        im_resz, lb_resz = inp.resize_image_label_with_min_check(image, label, 0.5)
        im_resz_v, lb_resz_v = self.sess.run([im_resz, lb_resz])
        self.failUnlessEqual(im_resz_v.ndim, 3)
        self.failUnlessEqual(im_resz_v.shape, lb_resz_v.shape)

        im_resz_chk, lb_resz_chk = inp.resize_image_label_with_min_check(
            image, label, 0.5, min_height=5, min_width=5)
        im_resz_chk_v, lb_resz_chk_v = self.sess.run([im_resz_chk, lb_resz_chk])
        self.failUnless(im_resz_chk_v.shape[0] >= 5)
        self.failUnless(im_resz_chk_v.shape[1] >= 5)
        self.failUnlessEqual(im_resz_chk_v.shape, lb_resz_chk_v.shape)

    def test_random_resize_image_label_with_min_check(self):
        image_v = np.zeros((8,10,3), dtype=np.uint8)
        label_v = np.zeros((8,10,3), dtype=np.uint8)
        image = ops.convert_to_tensor(image_v)
        label = ops.convert_to_tensor(label_v)
        im_resz, lb_resz = inp.random_resize_image_label(image, label, 0.5, 1.5,
                                                         min_height=5, min_width=5)
        for c in range(10):
            im_resz_v, lb_resz_v = self.sess.run([im_resz, lb_resz])
            self.failUnlessEqual(im_resz_v.ndim, 3)
            self.failUnlessEqual(im_resz_v.shape, lb_resz_v.shape)
            self.failUnless(im_resz_v.shape[0] >= 5)
            self.failUnless(im_resz_v.shape[1] >= 5)
            print "\t{}:{}".format(c, im_resz_v.shape)

    def test_load_image_label_list(self):
        fl_im = self.sess.run(self.input_builder.graph['im_files'])
        fl_lb = self.sess.run(self.input_builder.graph['lb_files'])
        for f, b, i in zip(fl_im, fl_lb, range(10)):
            self.failUnlessEqual(f[-3:],'png')
            self.failUnlessEqual(b[-3:],'png')
            self.failUnless(os.path.exists(f))
            self.failUnless(os.path.exists(b))

            if i % 5 == 0:
                logging.info("Image file:{}".format(f))
                logging.info("Label file:{}".format(b))

    def test_load_raw_image_label(self):
        im = self.input_builder.graph['raw_image']
        lb = self.input_builder.graph['raw_label']
        for c in range(2):
            im_v, lb_v = self.sess.run([im, lb])
            logging.info("Sample {}:{}".format(c, im_v.shape))

    def test_input_loader(self):
        ig = self.input_builder.graph
        for i in range(5):
            ims, lbs = self.sess.run([ig['image_batch'], ig['label_batch']])
            # assert ims and lbs has the same size as specified in configuration
            assert isinstance(ims, np.ndarray)
            assert isinstance(lbs, np.ndarray)
            self.failUnlessEqual(ims.ndim, 4)
            self.failUnlessEqual(lbs.ndim, 4)
            imh = self.conf['data']['jitter']['trn_im_height']
            imw = self.conf['data']['jitter']['trn_im_width']
            nc = self.conf['data']['num_classes']
            bs = self.conf['solver']['batch_size']
            self.failUnlessEqual( ims.shape, (bs, imh, imw, 3) )
            self.failUnlessEqual( lbs.shape, (bs, imh, imw, nc) )
            logging.info("Batch {}:{}:{}:{}".format(i, ims.shape, ims.dtype, lbs.dtype))

    def test_input_loader_vis(self):
        ig = self.input_builder.graph
        odir = os.path.join(
            self.conf['path']['base'],
            self.conf['path']['run'],
            self.conf['path']['this_experiment']
        )
        for b in range(5):
            ims, lbs = self.sess.run([ig['image_batch'], ig['label_batch']])
            bs = self.conf['solver']['batch_size']

            for i in range(bs):
                im = ims[i]
                lb = lbs[i]
                overlay_im = inp.overlay_image_with_class_colours(
                    im, lb,
                    class_colours=self.input_builder.get_class_colours())
                ofile = os.path.join(odir, "batch_{:02d}_sample_{:02d}.png".format(b, i))
                imsave(ofile, overlay_im)
                ofile = os.path.join(odir, "batch_{:02d}_sample_{:02d}_im.png".format(b, i))
                imsave(ofile, im)

    @unittest.skipUnless(TRIVIAL_TESTS, "trivial")
    def test_tf_random(self):
        f = tf.random_uniform((1,), 0.5, 1.5)
        for i in range(10):
            f_v = self.sess.run(f)
            print i, f_v

def main():
    unittest.main()

if __name__ == '__main__':
    main()
