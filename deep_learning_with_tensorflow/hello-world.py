import tensorflow as tf

hello = tf.constant('Hello, TensorFlow')

session = tf.Session()
print(session.run(hello))

a = tf.constant([2])
b = tf.constant([3])
c = tf.add(a, b)

with tf.Session() as session:
    result = session.run(c)
    print(result)
