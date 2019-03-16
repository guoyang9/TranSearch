from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys, time

import numpy as np 
import tensorflow as tf 
import data_input


class PreTrain(object):
	def __init__(self, visual_size, text_size, embed_size, 
							lr, dropout, is_training):
		"""
			For both visual and textual parts, we use
			several fully connnected layers,
			visual_size is 4096, textual_size is 512.
		"""
		self.visual_size = visual_size
		self.text_size = text_size
		self.embed_size = embed_size
		self.lr = lr
		self.dropout = dropout
		self.is_training = is_training

	def get_data(self, sample):
		self.anchor_text = sample[0]
		self.anchor_vis = sample[1]

		if self.is_training:
			self.pos_text = sample[2]
			self.pos_vis = sample[3]
			self.neg_text = sample[4]
			self.neg_vis = sample[5]
	
	def create_model(self):
		""" Create model from scratch. """
		def add_encode_layers(text, vis):
			with tf.name_scope("text_encode"):
				text = tf.layers.dense(text,
					activation=tf.nn.elu, reuse=tf.AUTO_REUSE,
					units=self.embed_size, name='text_forward_layer1')
				text = tf.layers.dropout(text, rate=self.dropout)
				text_forward = tf.layers.dense(text,
					activation=tf.nn.elu, reuse=tf.AUTO_REUSE,
					units=self.embed_size, name='text_forward_layer2')

			with tf.name_scope("vis_encode"):
				vis = tf.layers.dense(vis,
					activation=tf.nn.elu, reuse=tf.AUTO_REUSE,
					units=self.embed_size, name='vis_forward_layer1')
				vis = tf.layers.dropout(vis, rate=self.dropout)
				vis_forward = tf.layers.dense(vis,
					activation=tf.nn.elu, reuse=tf.AUTO_REUSE,
					units=self.embed_size, name='vis_forward_layer2')

			return text_forward, vis_forward

		def add_decode_layers(text_forward, vis_forward):
			with tf.name_scope("text_decode"):
				text = tf.layers.dense(text_forward,
					activation=tf.nn.elu, reuse=tf.AUTO_REUSE,
					units=self.embed_size, name='text_backward_layer1')
				text = tf.layers.dropout(text, rate=self.dropout)
				text = tf.layers.dense(text,
					activation=tf.nn.elu, reuse=tf.AUTO_REUSE,
					units=self.text_size, name='text_backward_layer2')

			with tf.name_scope("vis_decode"):
				vis = tf.layers.dense(vis_forward,
					activation=tf.nn.elu, reuse=tf.AUTO_REUSE,
					units=self.embed_size, name='vis_backward_layer1')
				vis = tf.layers.dropout(vis, rate=self.dropout)
				vis = tf.layers.dense(vis,
					activation=tf.nn.elu, reuse=tf.AUTO_REUSE,
					units=self.visual_size, name='vis_backward_layer2')

			return text, vis

		self.anchor_text_forward, self.anchor_vis_forward = add_encode_layers(
								self.anchor_text, self.anchor_vis)
		self.pos_text_forward, self.pos_vis_forward = add_encode_layers(
								self.pos_text, self.pos_vis)
		self.neg_text_forward, self.neg_vis_forward = add_encode_layers(
								self.neg_text, self.neg_vis)

		self.anchor_text_backward, self.anchor_vis_backward = add_decode_layers(
							self.anchor_text_forward, self.anchor_vis_forward)
		self.pos_text_backward, self.pos_vis_backward = add_decode_layers(
							self.pos_text_forward, self.pos_vis_forward)
		self.neg_text_backward, self.neg_vis_backward = add_decode_layers(
							self.neg_text_forward, self.neg_vis_forward)

		def triplet_loss(anchor, pos, neg, margin=1.0):
			# dis_pos = tf.norm(anchor - pos, axis=-1)
			# dis_neg = tf.norm(anchor - neg, axis=-1)
			dis_pos = tf.sqrt(tf.reduce_sum(tf.square(anchor-pos), -1) + 1e-6)
			dis_neg = tf.sqrt(tf.reduce_sum(tf.square(anchor-neg), -1) + 1e-6)

			loss = tf.maximum(0.0, margin + dis_pos - dis_neg)
			loss = tf.reduce_mean(loss)

			return loss

		with tf.name_scope("loss"):
			self.triplet_loss_text = triplet_loss(
				self.anchor_text_backward, self.pos_text_backward, self.neg_text_backward)
			self.triplet_loss_vis = triplet_loss(
				self.anchor_vis_backward, self.pos_vis_backward, self.neg_vis_backward)

			self.mse_loss_anchor_text = tf.losses.mean_squared_error(
								self.anchor_text_backward, self.anchor_text)
			self.mse_loss_pos_text = tf.losses.mean_squared_error(
								self.pos_text_backward, self.pos_text)
			self.mse_loss_neg_text = tf.losses.mean_squared_error(
								self.neg_text_backward, self.neg_text)

			self.mse_loss_anchor_vis = tf.losses.mean_squared_error(
								self.anchor_vis_backward, self.anchor_vis)
			self.mse_loss_pos_vis = tf.losses.mean_squared_error(
								self.pos_vis_backward, self.pos_vis)
			self.mse_loss_neg_vis = tf.losses.mean_squared_error(
								self.neg_vis_backward, self.neg_vis)

			self.final_loss = self.triplet_loss_text + self.triplet_loss_vis + 0.01 * (
				self.mse_loss_anchor_text + self.mse_loss_anchor_vis + 
				self.mse_loss_pos_text + self.mse_loss_pos_vis + 
				self.mse_loss_neg_text + self.mse_loss_neg_vis)

		with tf.name_scope("optimization"):
			self.optimizer = tf.train.AdamOptimizer(self.lr, name='Adam')
			self.optim = self.optimizer.minimize(self.final_loss)

	def summary(self):
		""" Create summaries to write on tensorboard. """
		self.writer = tf.summary.FileWriter('./graphs/pre_train', tf.get_default_graph())
		with tf.name_scope("summaries"):
			tf.summary.scalar('loss', self.final_loss)
			tf.summary.histogram('histogram loss', self.final_loss)
			self.summary_op = tf.summary.merge_all()

	def build(self):
		""" Build the computation graph. """
		# self.get_data()
		self.create_model()
		self.summary()
		self.saver = tf.train.Saver(tf.global_variables())

	def step(self, session, step):
		""" Train the model step by step. """
		if self.is_training:
			outputs = session.run([self.final_loss, self.optim, self.summary_op])
			self.writer.add_summary(outputs[-1], global_step=step)

def main(argv=None):

	FLAGS = tf.app.flags.FLAGS

	tf.app.flags.DEFINE_integer('batch_size', 512, 
				'size of mini-batch.')
	tf.app.flags.DEFINE_integer('negative_num', 20, 
				'number of negative samples.')
	tf.app.flags.DEFINE_integer('embedding_size', 128, 
				'the size for embedding user and item.')
	tf.app.flags.DEFINE_integer('epochs', 20, 
				'the number of epochs.')
	tf.app.flags.DEFINE_string('dataset', 'MenClothing', 
				'the pre-trained dataset.')
	tf.app.flags.DEFINE_string('model_dir', './Variables/', 
				'the dir for saving model.')
	tf.app.flags.DEFINE_string('gpu', '0', 
				'the gpu card number.')
	tf.app.flags.DEFINE_float('lr', 0.001, 
				'learning rate.')
	tf.app.flags.DEFINE_float('dropout', 0.5, 
				'dropout rate.')

	opt_gpu = FLAGS.gpu
	os.environ["CUDA_VISIBLE_DEVICES"] = opt_gpu
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True

	############################# PREPARE DATASET ###########################

	data = data_input.pretrainData(FLAGS.dataset, FLAGS.negative_num)
	train_dataset = tf.data.Dataset.from_generator(data.build,
				output_types=(tf.float32, tf.float32, tf.float32, 
							tf.float32, tf.float32, tf.float32),
				output_shapes=(tf.TensorShape([512]),
							tf.TensorShape([4096]),
							tf.TensorShape([512]),
							tf.TensorShape([4096]),
							tf.TensorShape([512]),
							tf.TensorShape([4096])))
	train_dataset = train_dataset.shuffle(100000).batch(FLAGS.batch_size)
	train_iter = train_dataset.make_initializable_iterator()

	with tf.Session(config=config) as sess:	
		sess.run(train_iter.initializer)
		train_next = train_iter.get_next()

	############################## CREATE MODEL #############################

		model = PreTrain(text_size=512, visual_size=4096, 
						embed_size=FLAGS.embedding_size, 
						lr=FLAGS.lr, dropout=FLAGS.dropout,is_training=True)		
		model.get_data(train_next)
		model.build()

		ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
		if ckpt:
			print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
			model.saver.restore(sess, ckpt.model_checkpoint_path)
		else:
			print("Creating model with fresh parameters.")
			sess.run(tf.global_variables_initializer())

	################################## TRAINING ############################

		count = 0
		for epoch in range(FLAGS.epochs):
			start_time = time.time()

			try:
				while True:
					model.step(sess, count)
					count += 1
			except tf.errors.OutOfRangeError:
				sess.run(train_iter.initializer)
				
				elapsed_time = time.time() - start_time
				print("Epoch: %d\t" %(epoch) + "Epoch time: " +
						time.strftime("%H: %M: %S", time.gmtime(elapsed_time))+'\n')

			train_next = train_iter.get_next()
			model.get_data(train_next)

	################################## SAVE MODEL ##########################

		checkpoint_path = os.path.join(FLAGS.model_dir, "pre_train.ckpt")
		model.saver.save(sess, checkpoint_path)


if __name__ == '__main__':
	tf.app.run()
