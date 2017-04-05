"""
Test if data feeder works fine
"""

import os
import shutil
import unittest
import json
import sys
import tensorflow as tf
import logging
from subprocess import Popen, PIPE

sys.path.insert(0, '..')
# noinspection PyPep8
import setup_test as sett

REDO_TRAIN = False
#  import numpy as np
#  from scipy.misc import imsave
#  from tensorflow.python.framework import dtypes, ops
#  import train_and_evaluate as tv

logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)s %(funcName)s: %(message)s')


class InputTests(unittest.TestCase):
    def setUp(self):
        with open('../config/test_config_000.json', 'r') as f:
            self.conf = json.load(f)

        tdir = os.path.join(
            self.conf['path']['base'],
            self.conf['path']['run'],
            self.conf['version']['id']
        )
        self.tdir = tdir

    def tearDown(self):
        tf.reset_default_graph()

    @unittest.skipUnless(REDO_TRAIN, "Don't redo training")
    def test_start_train(self):
        if os.path.exists(self.tdir):
            shutil.rmtree(self.tdir)
        sett.setup_run_dir(self.conf)
        cwd = r'{}/src'.format(self.tdir)
        subp = Popen(['python', 'train_and_evaluate.py', 'config_000.json', '--no-validation'],
                     env=os.environ.copy(), cwd=cwd, stderr=PIPE, stdout=PIPE)
        stdo_msg, stde_msg = subp.communicate()
        trn_succ_msg = "Training was completed successfully"
        val_succ_msg = "Validation was completed successfully"
        with open(os.path.join(self.tdir, 'test002_test_train_and_eval_log.txt'), 'w') as f:
            f.write("== STDOUT ==\n")
            f.write(stdo_msg)
            f.write("== STDERR ==\n")
            f.write(stde_msg)
        self.failUnless(trn_succ_msg in stdo_msg or trn_succ_msg in stde_msg)
        self.failUnless(val_succ_msg in stdo_msg or val_succ_msg in stde_msg)

    @unittest.skipIf(REDO_TRAIN, "Redo training and validation")
    def test_evaluate(self):
        cwd = r'{}/src'.format(self.tdir)
        subp = Popen(['python', 'train_and_evaluate.py', 'config_000.json', '--no-training'],
                     env=os.environ.copy(), cwd=cwd, stderr=PIPE, stdout=PIPE)
        stdo_msg, stde_msg = subp.communicate()
        val_succ_msg = "Validation was completed successfully"
        with open(os.path.join(self.tdir, 'test002_test_eval_log.txt'), 'w') as f:
            f.write("== STDOUT ==\n")
            f.write(stdo_msg)
            f.write("== STDERR ==\n")
            f.write(stde_msg)
        self.failUnless(val_succ_msg in stdo_msg or val_succ_msg in stde_msg)



def main():
    unittest.main()


if __name__ == '__main__':
    main()
