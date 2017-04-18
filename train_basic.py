import os
import numpy as np
import tensorflow as tf
import train_utils
import datetime

MATCHES_TRAINING = "matches_train.csv.gz"
MATCHES_TEST = "matches_test.csv.gz"

N_TRAINING_STEPS = 400000
ROWS_PER_BATCH = 1024

training_set = train_utils.infinitely_stream_gzipped_csv_without_header(
    filename=MATCHES_TRAINING, target_dtype=np.int, features_dtype=np.int)
test_set = train_utils.load_gzipped_csv_without_header(
    filename=MATCHES_TEST, target_dtype=np.int, features_dtype=np.int, max_rows=1000000)

N_FEATURES = 240

x = tf.placeholder(tf.float32, shape=[None, N_FEATURES])
y_ = tf.placeholder(tf.float32, shape=[None])

W = tf.Variable(tf.zeros([N_FEATURES, 1]))
b = tf.Variable(tf.zeros([1]))

y = tf.reshape(tf.nn.sigmoid(tf.tensordot(x,W,1)+b), [-1]) # reshape to 1-d

loss = tf.losses.mean_squared_error(y_, y)
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

correct_prediction = tf.equal(tf.round(y), tf.cast(y_, tf.float32))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

config = tf.ConfigProto()
# config.log_device_placement=True
config.gpu_options.per_process_gpu_memory_fraction=0.8

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    start = datetime.datetime.now()
    for step_index in range(N_TRAINING_STEPS):
        if step_index % 100 == 0:
            print 'on step {} out of {}'.format(step_index, N_TRAINING_STEPS)
            if step_index != 0:
                now = datetime.datetime.now()
                end = start + (now - start) * N_TRAINING_STEPS / step_index
                print 'expected end time: {}, which is {}s from now'.format(end.isoformat(), (end - now).total_seconds())
        batch_features, batch_targets = train_utils.next_training_batch(training_set, ROWS_PER_BATCH)
        train_step.run(feed_dict={x: batch_features, y_: batch_targets})
    print "Accuracy: "
    print(accuracy.eval(feed_dict={x: test_set[0], y_: test_set[1]}))
    saver = tf.train.Saver()
    saver.save(sess, 'basic-model')
