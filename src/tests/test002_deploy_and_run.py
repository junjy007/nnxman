"""
Test if data feeder works fine
"""

import os
import shutil
import unittest
import json
import sys
from multiprocessing import Process
sys.path.insert(0, '..')
import tensorflow as tf
from tensorflow.python.framework import dtypes, ops
import numpy as np
import logging
from scipy.misc import imsave
import setup_test as sett
import train_and_evaluate as tv
from subprocess import Popen, PIPE

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
        if os.path.exists(tdir):
            shutil.rmtree(tdir)
        sett.setup_run_dir(self.conf)
        self.tdir = tdir

    def tearDown(self):
        tf.reset_default_graph()

    def test_start_train(self):
        #cmd = r'python {}/src/train_and_evaluate.py --no-validation test_config_000.json'.format(self.tdir)
        cwd = r'{}/src'.format(self.tdir)
        #subp = Popen(['python', 'train_and_evaluate.py'], env=os.environ.copy(), cwd=cwd, stderr=PIPE, stdout=PIPE)
        subp = Popen(['python', 'train_and_evaluate.py', 'config_000.json', '--no-validation'],
                     env=os.environ.copy(), cwd=cwd, stderr=PIPE, stdout=PIPE)
        stdo_msg, stde_msg = subp.communicate()
        print stdo_msg
        print stde_msg
        # wait, then check first checkpoint


def main():
    unittest.main()

if __name__ == '__main__':
    main()
