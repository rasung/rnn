import tensorflow as tf
from tensorflow.contrib import rnn
import csv
import argparse
import os

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train-file',
        required=True
    )

    parser.add_argument(
        '--job-name',
        required=True
    )

    parser.add_argument(
        '--job-dir',
        required=True
    )
    args = parser.parse_args()
    arguments = args.__dict__

    pathToJobDir = arguments.pop('job_dir')
    jobName = arguments.pop('job_name')
    pathToData = arguments.pop('train_file')


    csv_file = pathToData
    hidden_size = 4
    batch_size = 1
    input_sequence_length = 30  
    output_sequence_length = 1
    input_num_classes = 18
    output_num_classes = 18
    stack = 5
    softmax_count = 3
    softmax_hidden_size = input_sequence_length

    X = tf.placeholder(tf.int32, [None, input_sequence_length])
    X_one_hot = tf.one_hot(X, input_num_classes)
    print("X_one_hot", X_one_hot)  # check out the shape

    Y = tf.placeholder(tf.int32, [None, output_sequence_length])  # 1
    Y_one_hot = tf.one_hot(Y, output_num_classes)  # one hot
    print("Y_one_hot", Y_one_hot)
    Y_one_hot = tf.reshape(Y_one_hot, [-1, output_num_classes])
    print("Y_reshape", Y_one_hot)


    # Make a lstm cell with hidden_size (each unit output vector size)
    def lstm_cell():
        cell = rnn.BasicLSTMCell(hidden_size, state_is_tuple=True)
        return cell

    multi_cells = rnn.MultiRNNCell([lstm_cell() for _ in range(stack)], state_is_tuple=True)

    # outputs: unfolding size x hidden size, state = hidden size
    outputs, _states = tf.nn.dynamic_rnn(multi_cells, X_one_hot, dtype=tf.float32)
    # reshape out
    outputs = tf.reshape(outputs, [batch_size, hidden_size * input_sequence_length])


    W1 = tf.Variable(tf.random_normal([hidden_size * input_sequence_length, softmax_hidden_size]), name='weight1')
    b1 = tf.Variable(tf.random_normal([softmax_hidden_size]), name='bias1')

    outputs = tf.matmul(outputs, W1) + b1

    W2 = tf.Variable(tf.random_normal([softmax_hidden_size, softmax_hidden_size]), name='weight2')
    b2 = tf.Variable(tf.random_normal([softmax_hidden_size]), name='bias2')

    outputs = tf.matmul(outputs, W2) + b2

    W3 = tf.Variable(tf.random_normal([softmax_hidden_size, output_num_classes]), name='weight3')
    b3 = tf.Variable(tf.random_normal([output_num_classes]), name='bias3')

    # tf.nn.softmax computes softmax activations
    logits = tf.matmul(outputs, W3) + b3
    hypothesis = tf.nn.softmax(logits)

    # Cross entropy cost/loss
    cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                    labels=Y_one_hot)
    cost = tf.reduce_mean(cost_i)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

    prediction = tf.argmax(hypothesis, 1)
    correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    filename_queue = tf.train.string_input_producer([csv_file])
    key, value = tf.TextLineReader().read(filename_queue)

    input_list = []
    for l in range(input_sequence_length + 1 + 1):
        input_list.append([1])

    data = tf.decode_csv(value, record_defaults=input_list)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        i=0
        while i < 6000000:
            i+=1
            datas = sess.run(data)
        
            dataX = []
            dataY = []
            # if testing max value = dataX.append(datas[0:-2]) dataY.append(datas[-2:-1])
            # else if testing min value = dataX.append(datas[0:-2]) dataY.append(datas[-1:])
            dataX.append(datas[0:-2])
            dataY.append(datas[-2:-1])

            print(i, dataX, dataY)
            
            _, loss, accur, hypo = sess.run([optimizer, cost, accuracy, hypothesis], feed_dict={X: dataX, Y: dataY})

            hypo_list=[]
            for j in range(len(hypo[0])):
                hypo_list.append(round(hypo[0][j], 2))

            print(loss, accur, hypo_list)

        coord.request_stop()
        coord.join(threads)

        saver = tf.train.Saver()
        model_file = os.path.join(pathToJobDir, jobName)
        saver.save(sess, model_file, global_step=0)
