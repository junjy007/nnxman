import tensorflow as tf

with tf.variable_scope("foo") as foo_scope:
    v = tf.get_variable("v", [1])
with tf.variable_scope(foo_scope):
    w = tf.get_variable("w", [1])
with tf.variable_scope(foo_scope, reuse=True):
    v1 = tf.get_variable("v", [1])
    w1 = tf.get_variable("w", [1])
print "v[{}]: name:'{}'".format(id(v),v.name)
print "w[{}]: name:'{}'".format(id(w),w.name)
print "v1[{}]: name:'{}'".format(id(v1),v1.name)
print "w1[{}]: name:'{}'".format(id(w1),w1.name)

with tf.variable_scope("sa") as my_scope:
    a = tf.placeholder(tf.float32,name="a")

with tf.variable_scope(my_scope, reuse=True):
    b = tf.placeholder(tf.float32,name="b")
    a1 = tf.placeholder(tf.float32,name="a")

with tf.variable_scope(my_scope, reuse=False):
    a2 = tf.placeholder(tf.float32,name="a")

print "a[{}]: name:'{}'".format(id(a),a.name)
print "b[{}]: name:'{}'".format(id(b),b.name)
print "a1[{}]: name:'{}'".format(id(a1),a1.name)
print "a2[{}]: name:'{}'".format(id(a2),a2.name)
print "a1<=>a: {}, a2<=>a: {}".format((a1 is a), (a2 is a))

