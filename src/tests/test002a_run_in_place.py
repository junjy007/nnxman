"""
Test running without deploying (so we can see the training output interactively)
"""

import os
import shutil
import unittest
import json
import sys
import tensorflow as tf
import logging

sys.path.insert(0, '..')
# noinspection PyPep8
import setup_test as sett
import train_and_evaluate as te

logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)s %(funcName)s: %(message)s')


class TrainTests(unittest.TestCase):
    def setUp(self):
        with open('../config/test_config_000.json', 'r') as f:
            self.conf = json.load(f)

        tmp_dir_name = os.path.splitext(os.path.split(__file__)[1])[0]
        self.conf['path']['this_experiment'] = 'rundir_' + tmp_dir_name
        tdir = os.path.join(
            self.conf['path']['base'],
            self.conf['path']['run'],
            self.conf['path']['this_experiment']
        )
        print tdir
        if os.path.exists(tdir):
            shutil.rmtree(tdir)
        sett.setup_run_dir(self.conf)
        self.tdir = tdir

    def tearDown(self):
        tf.reset_default_graph()

    def test_start_train(self):
        te.train(self.conf)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
