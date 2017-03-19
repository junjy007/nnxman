"""
Load VGG using tensorflow and extract features for an image

Usage:
    test_build_vgg.py build <net> 
    test_build_vgg.py variable <net> <layer> <vname> outfile <ofname>
    test_build_vgg.py forward <net> <layer> infile <ifname> outfile <ofname>

"""

import os
import sys

import numpy as np
import scipy.misc # for loading images
import tensorflow as tf

######## USING DOWNLOADED VGG ########
# -- I WANT TO DIY VGG - SO I CAN DO FURTHER ADJUST
#
#TFVGG_DIR="/home/junli/projects/tensorflow-fcn/KittiSeg/incl"
#TESTIMG_PATH="tmp_data/img1.jpg"
#sys.path.insert(1, TFVGG_DIR)
#from tensorflow_fcn import fcn16_vgg
#######################################

PROJECT_DIR=os.environ['PROJECT_DIR']
DATA_DIR=os.environ['DATA_DIR']
SAVE_DIR=os.environ['SAVE_DIR']
VGG_PRETRAINED_PARAM_FILE=\
        os.path.join(SAVE_DIR, 'vgg_pretrained', 'vgg16.npy')

REF_KITTISEG_PROJECT_DIR=os.path.join(PROJECT_DIR, 'ref', 'KittiSeg')
REF_TFFCN_PROJECT_DIR=os.path.join(REF_KITTISEG_PROJECT_DIR,
                                   'incl', 'tensorflow_fcn')

sys.path.append(REF_TFFCN_PROJECT_DIR)
from fcn16_vgg import FCN16VGG 
from src.sandbox.my_fcn16vgg import FullConvNet_VGG16

t_graph=tf.Graph()
with t_graph.as_default():
    tf.set_random_seed(1)
    t_input_image=tf.placeholder(tf.float32)
    t_single_input_image_batch=tf.expand_dims(t_input_image, 0)
    t_single_input_image_batch=tf.Print(t_single_input_image_batch,
            [tf.shape(t_single_input_image_batch)],
            message="Input image batch: ",
            summarize=4,
            first_n=1)


def load_image(img_file, img_width=400, img_height=300):

    img1 = scipy.misc.imread(img_file)
    img2 = scipy.misc.imresize(img1, [img_height, img_width])

    print "Image resize {}".format(img2.shape)
    return img2



#### Data prep done ####

def test2_build_my_net(netname):
    opts={'weight_decay_rate':0.01,
          'debug_info':{'load_weights':False}}
    v16=FullConvNet_VGG16(VGG_PRETRAINED_PARAM_FILE, opts)
    with t_graph.as_default():
        with tf.variable_scope(netname):
            v16.build(t_single_input_image_batch, debug=True)
    return v16

        # layer to be tested
        #out_v16=v16.score_pool4
        #ss = tf.Session()
        #tf.set_random_seed(1)
        #init=tf.global_variables_initializer()
        #ss.run(init)
        #my_net_out=ss.run(out_v16, {t_input_image: input_image})
        #ss.close()

    #return my_net_out

def test3_build_ref_net(netname):
    tf.reset_default_graph()
    ref_v16=FCN16VGG(VGG_PRETRAINED_PARAM_FILE)
    with t_graph.as_default():
        with tf.variable_scope(netname):
            ref_v16.build(t_single_input_image_batch)
    return ref_v16
        # layer to be tested
        #ref_net_out=ref_v16.score_pool4
        #ss = tf.Session()
        #tf.set_random_seed(1)
        #init=tf.global_variables_initializer()
        #ss.run(init)
        #out=ss.run(ref_net_out, {t_input_image:input_image})
        #ss.close()
    #return out

def test4_init_all():
    with t_graph.as_default():
        ss=tf.Session()
        init=tf.global_variables_initializer()
        ss.run(init)
    #train_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/train',
    #                                      sess.graph)
    #saver=tf.train.SummaryWriter(os.path.join(SAVE_DIR,"myvgg","build_graph"))
    return ss

def test5_get_variable(ss, netname, lname, vname):
    with t_graph.as_default():
        #vars=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='mynet')
        #print "Variables collection:{}".format(vars)
        #for v_ in vars:
        #    print "\t{}:[{}]".format(v_.name,v_.get_shape())
        with tf.variable_scope(netname, reuse=True):
            with tf.variable_scope(lname, reuse=True):
                v = tf.get_variable(vname)
                vv = ss.run(v)
    return vv

def test6_forward(ss, netobj, lname, input_image):
    ty = getattr(netobj,lname)
    return ss.run(ty, {t_input_image:input_image})
    
if __name__=="__main__":
    from docopt import docopt
    arguments=docopt(__doc__, version="0.0")

    if arguments['build']:
        if arguments['<net>']=='mynet':
            test2_build_my_net('mynet')
            test4_init_all()
        elif arguments['<net>']=='refnet':
            test3_build_ref_net('refnet')
            test4_init_all()
        else:
            raise ValueError
    elif arguments['variable']:
        if arguments['<net>']=='mynet':
            test2_build_my_net('mynet')
            ss=test4_init_all()
            v = test5_get_variable(ss, 'mynet', 
                    arguments['<layer>'],
                    arguments['<vname>'])
        elif arguments['<net>']=='refnet':
            test3_build_ref_net('refnet')
            ss=test4_init_all()
            v = test5_get_variable(ss, 'refnet', 
                    arguments['<layer>'],
                    arguments['<vname>'])
        else:
            raise ValueError
        ss.close()
        np.save(arguments['<ofname>'], v)
    elif arguments['forward']:
        input_image = load_image(arguments['<ifname>'])

        if arguments['<net>']=='mynet':
            v16net = test2_build_my_net('mynet')
        elif arguments['<net>']=='refnet':
            v16net = test3_build_ref_net('refnet')
        ss = test4_init_all()
        y = test6_forward(ss, v16net, arguments['<layer>'], input_image)
        np.save(arguments['<ofname>'], y)


    #my_out=test2_create_my_vgg16_net()
    #ref_out=test3_run_ref_fcn16vgg()
    #np.save('tmp_comp.npy', {'my_out':my_out, 'ref_out':ref_out})
