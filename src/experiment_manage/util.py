import tensorflow as tf
import os
import copy


def build_print_shape(pl, msg="", first_n=1):
    return tf.Print(pl, [tf.shape(pl)], message=msg, summarize=5, first_n=first_n)


def build_print_value(pl, msg="", summarise=5, first_n=1):
    return tf.Print(pl, [pl, ], message=msg, summarize=summarise, first_n=first_n)


def test_run(results, steps=1):
    R = []
    with tf.Session() as ss:
        init = tf.global_variables_initializer()
        ss.run(init)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=ss,
                                               coord=coord)

        for i in range(steps):
            R.append(ss.run(results))

        coord.request_stop()
        coord.join(threads)
        ss.close()
    return R


def copy_dict_part(tgt_dict, src_dict, key_list=None):
    """
    Copy some entries from the source list to the target one.
    :param tgt_dict:
    :param src_dict:
    :param key_list: a list has tuple entries as (old_key[, new_key])
    :return:
    """
    key_list = key_list or [(k,) for k in src_dict.keys()]
    for k in key_list:
        if len(k) > 1:
            old_k, new_k = k
        else:
            old_k = k[0]
            new_k = old_k
        tgt_dict[new_k] = copy.deepcopy(src_dict[old_k])


def get_experiment_running_dir(path_setting, exp_id):
    base_dir = path_setting['base']

    d0_ = path_setting['run']
    if not os.path.isabs(d0_):
        d0_ = os.path.join(base_dir, d0_)

    d_ = path_setting['this_experiment']
    if d_ == '#same-as-id':
        d_ = exp_id

    if len(d_)>0 and d_ != '#none':
        run_dir = os.path.join(d0_, d_)
    else:
        run_dir = d0_

    return run_dir
