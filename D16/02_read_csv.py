"""
Text file reading
"""
import tensorflow as tf
import os

def read_csv(filelist):
    # Building a file queue
    file_queue = tf.train.string_input_producer(filelist)
    # Building the reader
    reader = tf.TextLineReader()
    # Using a reader to read data in a file queue
    k, val = reader.read(file_queue)
    # Decode: convert to tensor
    records = [['None'], ['None']]
    x, y = tf.decode_csv(val, records)
    # batch file
    x_bat, y_bat = tf.train.batch([x, y],
                                  batch_size=8,    # Batch size
                                  num_threads=1)   # Number of threads
    return x_bat, y_bat


if __name__ == '__main__':
    # Build file list
    dir_name = './test_data/'
    file_names = os.listdir(dir_name)
    file_list = []
    for i in file_names:
        file_list.append(os.path.join(dir_name, i))
    x, y = read_csv(file_list)
    # Open Session, execute
    with tf.Session() as sess:
        # thread coordinator
        coord = tf.train.Coordinator()
        # Threads to open a queue to run
        threads = tf.train.start_queue_runners(sess, coord=coord)
        x_res, y_res = sess.run([x, y])
        print(x_res)
        print(y_res)

        # Wait for the thread to finish and reclaim resources
        coord.request_stop()
        coord.join(threads)

"""
[b'CCCCCCCCCC1' b'CCCCCCCCCC2' b'CCCCCCCCCC3' b'CCCCCCCCCC4'
 b'CCCCCCCCCC5' b'AAAAAAAAAAAA1' b'AAAAAAAAAAAA2' b'AAAAAAAAAAAA3']
[b'C1' b'C2' b'C3' b'C4' b'C5' b'A1' b'A2' b'A3']
"""