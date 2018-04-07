import tensorflow as tf

state = tf.Variable(0)
one = tf.constant(1)
new_value = tf.add(state, one)

update = tf.assign(state, new_value)

init_op = tf.initialize_all_variables()

with tf.Session() as session:
    session.run(init_op)
    print(session.run(state))
    for _ in range(3):
        print(session.run(update))
        print(session.run(state))
