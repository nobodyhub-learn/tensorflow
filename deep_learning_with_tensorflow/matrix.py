import tensorflow as tf

matrix1 = tf.constant([[1, 2, 3],
                       [2, 3, 4],
                       [3, 4, 5]])
matrix2 = tf.constant([[2, 2, 2],
                       [2, 2, 2],
                       [2, 2, 2]])
op1 = tf.add(matrix1, matrix2)
op2 = matrix1 + matrix2
with tf.Session() as session:
    print(session.run(op1))
    print(session.run(op2))

matrix_one = tf.constant([[2, 3],
                          [3, 4]])
matrix_two = tf.constant([[2, 3],
                          [3, 4]])
operation = tf.matmul(matrix_one, matrix_two)

with tf.Session() as session:
    result = session.run(operation)
print(result)
