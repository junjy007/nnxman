"""
Usage:
  setup_test <config-file> [--base-dir=<bdir>] [--run-dir=<rdir>] [--data-dir=<ddir>]

Options:
  --base-dir=<bdir>  Overriding the base-path specified in the experiment
    configuration file
  --run-dir=<rdir>  Overriding the running directory. Note there won't be
    experiment ID suffix to the directory, regardless whether it is
    a relative directory.
  --data-dir=<ddir>  Similar to run-dir above.
"""
import json
import os
import docopt
import shutil
import logging
from experiment_manage.util import get_experiment_running_dir


def get_dirs(conf_base_dir, args_base_dir,
             conf_run_dir, args_run_dir,
             conf_data_dir, args_data_dir):
    """
    In config file, dirs for :
    :param conf_base_dir: base
    :param conf_run_dir: run
    :param conf_data_dir: data
    In script arguments
    :param args_base_dir: base
    :param args_run_dir: run
    :param args_data_dir: data
    """
    base_dir = args_base_dir or conf_base_dir
    run_dir = args_run_dir or conf_run_dir
    data_dir = args_data_dir or conf_data_dir

    absdir = os.path.isabs
    pjoin = os.path.join

    assert absdir(base_dir), "Base dir must be absolute"

    if absdir(run_dir) and absdir(data_dir):
        base_dir = ""
    elif absdir(run_dir):
        data_dir = pjoin(base_dir, data_dir)
        base_dir = ""
    elif absdir(data_dir):
        run_dir = pjoin(base_dir, run_dir)
        base_dir = ""
    # If one of the run / data dir is absolute, we don't
    # keep a base dir to less confusion, and join it to
    # make the other absolute as well

    return {
        'base': base_dir,
        'run': run_dir,
        'data': data_dir,
    }


# noinspection PyShadowingNames
def setup_run_dir(conf):
    """
    :param conf: Dictionary read from the experiment JSON file.
    """
    pth = conf['path']
    base_dir = conf['path']['base']
    run_dir = get_experiment_running_dir(pth, conf['version']['id'])
    pjoin = os.path.join

    if not os.path.exists(run_dir):
        os.mkdir(run_dir)

    if os.path.exists(pjoin(run_dir, 'SETUP_DONE')):
        error_msg = \
            "Experiment directory \n\t{}\n already setup. To re-setup, " \
            "remove the SETUP_DONE file and re-run. Alternatively, you can " \
            "use the --run-dir=<rdir> option to assign a new running " \
            "directory (over-ruling the default one in the configuration " \
            "file.)".format(run_dir)
        raise Exception(error_msg)

    with open(pjoin(run_dir, 'SETUP_DONE'), 'w') as f:
        f.write("Done.")

    for dn_ in ['checkpoint_dir', 'train_summary_dir', 'valid_summary_dir']:
        if not os.path.exists(pjoin(run_dir, pth[dn_])):
            os.mkdir(pjoin(run_dir, pth[dn_]))

    # copy the model definitions
    src_files_dir = pjoin(run_dir, 'src')
    if not os.path.exists(src_files_dir):
        os.mkdir(src_files_dir)

    for c in conf['model'].keys():
        mod_path = conf['model'][c]
        src = pjoin(base_dir, mod_path)
        fname = os.path.split(src)[1]
        dst = pjoin(src_files_dir, fname)
        shutil.copy2(src, dst)
        conf['model'][c] = fname

    conf['version']['running_copy'] = True
    with open(pjoin(src_files_dir, 'config_000.json'), 'w') as f:
        json.dump(conf, f)

    # copy the framework
    my_dir = os.path.dirname(__file__)

    src_dir = pjoin(my_dir, 'experiment_manage')
    tgt_dir = pjoin(src_files_dir, 'experiment_manage')
    shutil.copytree(src_dir, tgt_dir)
    logging.debug("Copy framework from {} to {}".format(src_dir, tgt_dir))
    # copy run_test.py to the script directory
    shutil.copy2(pjoin(os.path.dirname(__file__), 'train_and_evaluate.py'),
                 pjoin(src_files_dir, 'train_and_evaluate.py'))
    return


if __name__ == "__main__":
    args = docopt.docopt(__doc__)
    with open(args['<config-file>'], 'r') as f:
        conf = json.load(f)

    dirs = get_dirs(conf_base_dir=conf['path']['base'],
                    args_base_dir=args['--base-dir'],
                    conf_run_dir=conf['path']['run'],
                    args_run_dir=args['--run-dir'],
                    conf_data_dir=conf['path']['data'],
                    args_data_dir=args['--data-dir'])
    for k, v in dirs.iteritems():
        conf['path'][k] = v

    # determine the running directory of the experiment
    setup_run_dir(conf)
