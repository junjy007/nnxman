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
import os
from multiprocessing import Process, Queue
from experiment_manage.core import ModelFactory
from experiment_manage.util import \
    CheckpointEvaluationRecord, \
    get_experiment_running_dir

logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)s %(filename)s: %(message)s')

TRAINING_COMPLETE_MSG = 'DONE'


def validation_d(conf, q):
    """
    :param conf:
    :param q: comm-queue, for training process to stop this validation process
    :return:
    """

    # noinspection PyShadowingNames
    def evaluate_checkpoint(conf, cp_path, rec_func):
        md = ModelFactory(conf)
        visualise_image_save_dir = os.path.splitext(cp_path)[0] \
                                   + '.visualise_validation'
        logging.info("cp-path:{}\n"
                     "visualisation dir: {}".format(cp_path, visualise_image_save_dir))
        loss = md.evaluate(cp_path = cp_path,
                           visimg_savedir=visualise_image_save_dir)[1] # mean-cross entropy
        rec_func(cp_path, loss)

    run_dir = get_experiment_running_dir(conf['path'], conf['version']['id'])
    checkpoint_dir = os.path.join(run_dir, conf['path']['checkpoint_dir'])
    eval_record = CheckpointEvaluationRecord(checkpoint_dir,
                                             conf['path']['checkpoint_name'],
                                             conf['path']['valid_record'])
    one_more = False
    while True:
        for p in eval_record.get_to_eval_checkpoint_paths():
            logging.info("Evaluate checkpoint {}".format(p))
            evaluate_checkpoint(conf, cp_path=p,
                                rec_func=eval_record.record_checkpoint_eval_result)
        if one_more:
            break
        # noinspection PyBroadException
        try:
            msg = q.get(timeout=120)
            if msg == TRAINING_COMPLETE_MSG:
                logging.info("Training was completed successfully. "
                             "Check saved model to eval once more.")
                one_more = True
        except:
            pass
    logging.info("Validation was completed successfully")


def train(conf, q=None):
    md = ModelFactory(conf)
    md.train()
    if q:
        q.put(TRAINING_COMPLETE_MSG)
    logging.info("Training was completed successfully.")


def main(config_file, do_training, do_validation):
    with open(config_file, 'r') as f:
        conf = json.load(f)

    q = Queue()
    if do_training:
        p_train = Process(target=train, args=(conf, q))
        p_train.start()
    if do_validation:
        p_valid = Process(target=validation_d, args=(conf, q))
        p_valid.start()

    if do_validation and not do_training:
        q.put(TRAINING_COMPLETE_MSG)

    if do_training:
        # noinspection PyUnboundLocalVariable
        p_train.join()
    if do_validation:
        # noinspection PyUnboundLocalVariable
        p_valid.join()

    return


if __name__ == '__main__':
    args = docopt.docopt(__doc__)

    main(args['<config_file>'],
         do_training=(not args['--no-training']),
         do_validation=(not args['--no-validation']))
