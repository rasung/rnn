import tensorflow as tf
from tensorflow.contrib import rnn
import csv

CSV_DATA_PATH_LIST = '1month_1month_19970101_20170831.csv'

f_data = open(CSV_DATA_PATH_LIST, 'r', encoding='utf-8')
rdr = csv.reader(f_data)

hidden_size = 10
batch_size = 1
input_sequence_length = 30  
output_sequence_length = 1
input_num_classes = 18
output_num_classes = 18

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

multi_cells = rnn.MultiRNNCell([lstm_cell() for _ in range(2)], state_is_tuple=True)

# outputs: unfolding size x hidden size, state = hidden size
outputs, _states = tf.nn.dynamic_rnn(multi_cells, X_one_hot, dtype=tf.float32)
# reshape out
outputs = tf.reshape(outputs, [batch_size, hidden_size * input_sequence_length])


W = tf.Variable(tf.random_normal([hidden_size * input_sequence_length, output_num_classes]), name='weight')
b = tf.Variable(tf.random_normal([output_num_classes]), name='bias')

# tf.nn.softmax computes softmax activations
logits = tf.matmul(outputs, W) + b
hypothesis = tf.nn.softmax(logits)

# Cross entropy cost/loss
cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                 labels=Y_one_hot)
cost = tf.reduce_mean(cost_i)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for line in rdr:
        dataX = []
        dataY = []
    
        # if testing max value = dataX.append(line[0:-2]) dataY.append(line[-2:-1])
        # else if testing min value = dataX.append(line[0:-2]) dataY.append(line[-1:])
        dataX.append(line[0:-2])
        dataY.append(line[-2:-1])

        print(dataX, dataY)
        
        _, loss, accur = sess.run([optimizer, cost, accuracy], feed_dict={X: dataX, Y: dataY})
        print(loss, accur)