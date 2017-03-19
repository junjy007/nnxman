import tensorflow as tf

x=tf.placeholder(tf.float32)
y=tf.placeholder(tf.float32)

z=x*y

with tf.Session() as ss:
  ss.run(tf.global_variables_initializer())
  print ss.run(z, feed_dict={x:[[1,2,3],[4,5,6]], y:[[1,1,1],[2,2,2]]})
  print ss.run(z, feed_dict={x:[[1,2,3],[4,5,6]], y:[2,2,2]})


