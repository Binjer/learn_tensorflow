import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import mnist_inference
import mnist_train

# 每10秒加载一次最新的模型，并在测试数据集上测试最新模型的正确率
EVAL_INTERVAL_SECS = 10


def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')

        y = mnist_inference.inference(x, None)

        correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
        accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # 通过变量重命名的方式来加载模型
        variable_averages = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        validation_feed = {x: mnist.validation.images, y_: mnist.validation.labels}

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

                global_step = ckpt.model_checkpoint_path.split("/")[-1].split("-")[-1]
                accuracy_score = sess.run(accuracy_op, feed_dict=validation_feed)

                print("After training %s step(s), validation accuracy is %g." % (global_step, accuracy_score))
            else:
                print("No checkpoint file found")
                return


def main(argv=None):
    mnist = input_data.read_data_sets("./data/MNIST_data", one_hot=True)
    evaluate(mnist)


if __name__ == "__main__":
    tf.app.run()
