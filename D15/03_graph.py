"""
Graph
"""
import tensorflow as tf

x = tf.constant(100.0)
y = tf.constant(200.0)
temp = tf.add(x, y)

# Get the default graph
graph = tf.get_default_graph()
print("Default graph: ", graph)

# New graph
new_graph = tf.Graph()
print("New graph: ", new_graph)

# Temporarily set the newly created graph as the default graph
with new_graph.as_default():
    new_op = tf.constant("hello kitty")

# Execution
with tf.Session(graph=new_graph) as sess:
    print(sess.run(new_op))

with tf.Session(graph=graph) as sess:
    print(sess.run(temp))