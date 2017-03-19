"""
Usage:
    train_and_evaluate.py [options] <config_file>

Options:
    --no-validation   if set, no validation will be performed, otherwise,
                      validation will be done when a checkpoint is saved
                      during training.
    --no-training     if set, only validation will be done.
"""
import json
import docopt
import logging
from multiprocessing import Process
from experiment_manage.core import ModelFactory

logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)s %(filename)s: %(message)s')

def main(config_file, do_training, do_validation):
    with open(config_file, 'r') as f:
        conf = json.load(f)

    md = ModelFactory(conf)

    def dummy(): pass
    fn_train = do_training and md.train or dummy
    fn_valid = do_validation and md.evaluate or dummy

    p_train = Process(target=fn_train)
    p_valid = Process(target=fn_valid)
    p_train.start()
    p_valid.start()
    p_train.join()
    p_valid.join()

    return


if __name__ == '__main__':
    args = docopt.docopt(__doc__)

    main(args['<config_file>'],
         do_training=(not args['--no-training']),
         do_validation=(not args['--no-validation']))
