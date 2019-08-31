import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import mnist_lenet_inference
import mnist_lenet_train


def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [5000, mnist_lenet_inference.IMAGE_SIZE,
                                        mnist_lenet_inference.IMAGE_SIZE,
                                        mnist_lenet_inference.NUM_CHANNELS], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, mnist_lenet_inference.OUTPUT_NODE], name='y-input')

        y = mnist_lenet_inference.inference(x, False, None)

        correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
        accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # 通过变量重命名的方式来加载模型
        variable_averages = tf.train.ExponentialMovingAverage(mnist_lenet_train.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        xv = np.reshape(mnist.validation.images,
                        (-1, mnist_lenet_inference.IMAGE_SIZE,
                         mnist_lenet_inference.IMAGE_SIZE,
                         mnist_lenet_inference.NUM_CHANNELS))
        yv = mnist.validation.labels

        validation_feed = {x: xv, y_: yv}

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(mnist_lenet_train.MODEL_SAVE_PATH)
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
