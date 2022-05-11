# from https://github.com/shivsondhi/Triplet-Loss/blob/master/triplet_loss_functions.py
import tensorflow as tf

def triplet_loss_fn(x, alpha=0.3):
	anchor, positive, negative = x

	positive_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
	negative_dist = tf.reduce_sum(tf.square(tf.subtract(anchor,negative)), 1)

	loss_1 = tf.add(tf.subtract(positive_dist, negative_dist), alpha)
	loss = tf.reduce_sum(tf.maximum(loss_1, 0.0), 0)

	return loss