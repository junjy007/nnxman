import numpy as np
import unittest
import sys
import logging

sys.path.insert(0, '..')
# noinspection PyPep8
import experiment_manage.util as exutil

logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)s %(funcName)s: %(message)s')


class InputTests(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_convert_class_idx_to_onehot(self):
        y = np.asarray([0, 1, 2])
        y_code = exutil.class_idx_to_onehot(y)
        y_code1 = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                             dtype=np.bool)
        self.failUnless(np.all(y_code == y_code1))
        y = np.asarray([[0, 1, 2], [0, 0, 1]])
        y_code = exutil.class_idx_to_onehot(y)

        y_code_c1 = np.asarray([[1, 0, 0], [1, 1, 0]], dtype=np.bool)
        y_code_c2 = np.asarray([[0, 1, 0], [0, 0, 1]], dtype=np.bool)
        y_code_c3 = np.asarray([[0, 0, 1], [0, 0, 0]], dtype=np.bool)
        y_code_c = np.stack((y_code_c1, y_code_c2, y_code_c3), axis=2)
        self.failUnless(np.all(y_code == y_code_c))


def main():
    unittest.main()


if __name__ == '__main__':
    main()
