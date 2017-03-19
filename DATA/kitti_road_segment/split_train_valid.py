"""
Usage:
    split_train_valid.py [--validation-percent=<vp>] [--randseed=<rs>] <all_sample_list> \
<train_sample_list> <valid_sample_list>

Options:
    --validation-percent=<vp>  the percentage of validation samples [default: 20]
    --randseed=<rs>            random seed [default: 42]
"""
import docopt
import numpy as np


def load_list(fn):
    with open(fn, 'r') as f:
        sample_list = [l for l in f]

    return sample_list


def split_list(l0, p, rs=None):
    """
    Split l0 into two lists, with the second one has a portion of p.
    :param l0: list to split
    :param p: proportion of 2nd list
    :param rs: random seed
    :return:
    """
    n = len(l0)
    n2 = int(n * p)
    rs = rs or 42
    rng = np.random.RandomState(rs)
    ind = list(range(n))
    rng.shuffle(ind)
    ind2 = ind[:n2]
    ind1 = ind[n2:]
    l1 = [l0[i] for i in ind1]
    l2 = [l0[i] for i in ind2]
    return l1, l2


def make_train_valid_sample_list(all_list_file, train_list_file, valid_list_file,
                                 valid_p, rs):
    l0 = load_list(all_list_file)
    l1, l2 = split_list(l0, valid_p, rs)
    with open(train_list_file, 'w') as f:
        for l in l1:
            f.write(l)
    with open(valid_list_file, 'w') as f:
        for l in l2:
            f.write(l)


if __name__ == "__main__":
    args = docopt.docopt(__doc__)

    make_train_valid_sample_list(args['<all_sample_list>'],
                                 args['<train_sample_list>'],
                                 args['<valid_sample_list>'],
                                 float(args['--validation-percent']) / 100,
                                 int(args['--randseed']))
