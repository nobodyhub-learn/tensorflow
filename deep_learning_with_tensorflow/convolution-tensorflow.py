import tensorflow as tf

input = tf.Variable(tf.random_normal([1, 10, 10, 1]))
filter = tf.Variable(tf.random_normal([3, 3, 1, 1]))

op = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID')
op2 = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')

init = tf.initialize_all_variables()
with tf.Session() as session:
    session.run(init)
    print('Input: \n {0} \n'.format(input.eval()))
    print('Filter: \n {0} \n'.format(filter.eval()))
    print('Result.Featre Map with valid position: \n {0} \n'.format(
        session.run(op)))
    print('Result.Featre Map with valid padding: \n {0} \n'.format(
        session.run(op2)))
