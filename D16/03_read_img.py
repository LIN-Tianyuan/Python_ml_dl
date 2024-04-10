"""
Image file reading
"""
import tensorflow as tf
import os

def read_img(filelist):
    # Building a file queue
    file_queue = tf.train.string_input_producer(filelist)
    # Building the reader
    reader = tf.WholeFileReader()
    # Using a reader to read data in a file queue
    k, val = reader.read(file_queue)
    # Decode: convert to tensor
    img = tf.image.decode_jpeg(val)
    # Batch processing (uniform size)
    img_resized = tf.image.resize(img, (250, 250))
    img_resized.set_shape([250, 250, 3])

    img_bat = tf.train.batch([img_resized],
                             batch_size=10,
                             num_threads=1)
    return img_bat


if __name__ == '__main__':
    # Build file list
    dir_name = './test_img/'
    file_names = os.listdir(dir_name)
    file_list = []
    for i in file_names:
        file_list.append(os.path.join(dir_name, i))
    imgs = read_img(file_list)
    # Open Session, execute
    with tf.Session() as sess:
        # thread coordinator
        coord = tf.train.Coordinator()
        # Threads to open a queue to run
        threads = tf.train.start_queue_runners(sess, coord=coord)
        img_res = sess.run([imgs])
        print(img_res)

        # Wait for the thread to finish and reclaim resources
        coord.request_stop()
        coord.join(threads)

