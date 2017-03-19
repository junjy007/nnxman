import tensorflow as tf
import numpy as np

XV=np.asarray([1,2,3])
def test1():
    """
    without specifying graph
    """
    g=tf.Graph()
    with g.as_default():
        with tf.variable_scope('tmp'):
            init=tf.constant_initializer(value=XV,dtype=tf.float32)
            x=tf.get_variable(name="x",initializer=init,shape=XV.shape)
            tf.add_to_collection('A', tf.reduce_sum(x))

            x=tf.Print(x,[tf.shape(x)],message="Test msg: ", 
                    summarize=2, first_n=1)
            ss=tf.Session()
            ginit=tf.global_variables_initializer()
            ss.run(ginit)
            xout=ss.run(x)
    return xout

def test2():
    """
    specifying graph
    """
    g=tf.Graph()
    with g.as_default():
        with tf.variable_scope('tmp'):
            init=tf.constant_initializer(value=XV,dtype=tf.float32)
            x=tf.get_variable(name="x",initializer=init,shape=XV.shape)

            tf.add_to_collection('A', tf.reduce_sum(x))

            x=tf.Print(x,[tf.shape(x)],message="Test msg: ", 
                    summarize=2, first_n=1)
            ss=tf.Session()
            ginit=tf.global_variables_initializer()
            ss.run(ginit)
            xout=ss.run(x)
    return xout


if __name__=='__main__':
    print test1()
    print test2()
    
