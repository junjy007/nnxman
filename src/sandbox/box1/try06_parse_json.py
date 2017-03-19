"""
Usage:
  setup_test <config-file> [--project-dir=<pdir>] [--run-dur=<rdir>]

Options:
  --project-dir=<pdir> Overriding the base-path in environmental variable
    $PROJECT_DIR, from which the relative pathes in the experiment configuration
    are constructed.
  --run-dir=<rdir> Overriding the running directory in config file. Note the
    parent directory of run-dir must exist.
"""

import json
import os
import docopt
import shutil

def determine_run_dir(config_run_dir, given_run_dir, id):
  """
  :param config_run_dir: run dir given in the config file
  :param given_run_dir: run dir given in invoking this program,
    over-ruling the config_run_dir
  :param id: experiment id, for automatic generating run_dir
  :return: run_dir in use
  """
  global PROJECT_DIR
  if given_run_dir:
    return given_run_dir
  # Rule: if the run-dir is a relative one, it is calculated from the
  # project directory, and will add a suffix of the experiment ID. Otherwise,
  # use it directly.
  if os.path.isabs(config_run_dir):
    return config_run_dir

  return os.path.join(PROJECT_DIR, config_run_dir, id)

def setup_run_dir(
    config,
    given_run_dir=None
):
  """
  :brief setup_run_dir do what ...

  :param config: Dictionary read from the experiment JSON file.
  :param run_dir2: A running directory provided when invoking this program,
   over-ruling the one in config
  :return:
  """
  global PROJECT_DIR
  run_dir = determine_run_dir(config['path']['run_dir'],
                              given_run_dir,
                              config['id_str'])
  config['path']['run_dir']=run_dir # to save in experiment running dir

  if not os.path.exists(run_dir): os.mkdir(run_dir)

  if os.path.exists(os.path.join(run_dir, 'SETUP_DONE')):
    error_msg = \
      "Experiment directory \n\t{}\n already setup. To re-setup, " \
      "remove the SETUP_DONE file and re-run. Alternatively, you can " \
      "use the --run-dir=<rdir> option to assign a new running " \
      "directory (over-ruling the default one in the configuration " \
      "file.)".format(run_dir)
    raise Exception(error_msg)

  # copy the model definitions
  script_dir = os.path.join(run_dir, 'scripts')
  if not os.path.exists(script_dir): os.mkdir(script_dir)

  for c in config['model'].keys():
    mod_path=config['model'][c]
    src = os.path.join(PROJECT_DIR, mod_path)
    fname = os.path.split(src)[1]
    dst = os.path.join(script_dir, fname)
    shutil.copy2(src, dst)
    config['model'][c]=fname

  with open(os.path.join(script_dir,'config.json'),'w') as f:
    json.dump(config,f)
  return


if __name__=="__main__":
  global PROJECT_DIR

  args = docopt.docopt(__doc__)

  if not args['--project-dir']:
    PROJECT_DIR = os.environ['PROJECT_DIR']
  else:
    PROJECT_DIR = args['--project-dir']

  if os.path.isabs(args['<config-file>']):
    conffile=args['<config-file>']
  else:
    conffile=os.path.join(PROJECT_DIR, args['<config-file>'])

  with open(conffile,'r') as f:
    config=json.load(f)

  if args['--run-dir']:
    setup_run_dir(config, args['--run-dir'])
  else:
    setup_run_dir(config)
