#FLOPS
# https://stackoverflow.com/questions/45085938/tensorflow-is-there-a-way-to-measure-flops-for-a-model

import tensorflow as tf
from tensorflow.python.framework import graph_util

def load_pb(pb):
    with tf.compat.v1.gfile.GFile(pb, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
        return graph

# # ***** (1) Create Graph *****
# g = tf.Graph()
# sess = tf.compat.v1.Session(graph=g)
# with g.as_default():
#     A = tf.Variable(initial_value=tf.compat.v1.random_normal([25, 16]))
#     B = tf.Variable(initial_value=tf.compat.v1.random_normal([16, 9]))
#     C = tf.matmul(A, B, name='output')
#     sess.run(tf.compat.v1.global_variables_initializer())
#     flops = tf.compat.v1.profiler.profile(g, options = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation())
#     print('FLOP before freezing', flops.total_float_ops)
# # # *****************************        

# # ***** (2) freeze graph *****
# output_graph_def = graph_util.convert_variables_to_constants(sess, g.as_graph_def(), ['output'])

# with tf.compat.v1.gfile.GFile('/models/prefabModels/md_v4.1.0.pb', "wb") as f:
#     f.write(output_graph_def.SerializeToString())
# # *****************************


# ***** (3) Load frozen graph *****
g2 = load_pb('./models/prefabModels/md_v4.1.0.pb')
with g2.as_default():
    flops = tf.compat.v1.profiler.profile(g2, options = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation())
    print('FLOP after freezing', flops.total_float_ops)