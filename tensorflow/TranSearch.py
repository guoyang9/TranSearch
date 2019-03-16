from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf 


class TranSearch(object):
	def __init__(self, visual_size, text_size, embed_size, item_size, 
					user_size, mode, lr, regularizer, optim, activation_func, 
					test_first, dropout, neg_num, is_training):
		"""
		Important Args:
		visual_size: the image feature size, is 4096 as default.
		text_size: the learned text size, default is 512.
		item_size: the number of items.
		user_size: number of users.
		mode: can be 'end', 'vis', 'text', 'double'.
		test_first: first compute all the item embeddings when testing.
		neg_num: the number of negative samples when training.
		"""
		self.visual_size = visual_size
		self.text_size = text_size
		self.embed_size = embed_size
		self.item_size = item_size
		self.user_size = user_size
		self.mode = mode 
		self.lr = lr
		self.regularizer_rate = regularizer
		self.activation_func = activation_func
		self.optim = optim
		self.test_first = test_first
		self.dropout = dropout
		self.neg_num = neg_num
		self.is_training = is_training

	def inference(self):
		""" Initialize important settings """
		with tf.name_scope("inputs"):
			self.user = tf.placeholder(dtype=tf.int32, 
								shape=[None], 
								name='user')
			self.query = tf.placeholder(dtype=tf.float32, 
								shape=[None, 512], 
								name='query')
			self.pos_text = tf.placeholder(dtype=tf.float32, 
								shape=[None, 512], 
								name='pos_text')
			self.pos_vis = tf.placeholder(dtype=tf.float32, 
								shape=[None, 4096], 
								name='pos_vis')
			self.neg_text = tf.placeholder(dtype=tf.float32, 
								shape=[None, self.neg_num, 512], 
								name='neg_text')
			self.neg_vis = tf.placeholder(dtype=tf.float32, 
								shape=[None, self.neg_num, 4096], 
								name='neg_vis')
			self.all_items = tf.placeholder(dtype=tf.float32, 
								shape=[None, self.embed_size],
								name='all_items')

		self.regularizer = tf.contrib.layers.l2_regularizer(
								self.regularizer_rate)
		if self.activation_func == 'ReLU':
			self.activation_func = tf.nn.relu
		elif self.activation_func == 'Leaky_ReLU':
			self.activation_func = tf.nn.leaky_relu
		elif self.activation_func == 'ELU':
			self.activation_func = tf.nn.elu

		if self.optim == 'SGD':
			self.optimizer = tf.train.GradientDescentOptimizer(self.lr, 
								name='SGD')
		elif self.optim == 'RMSProp':
			self.optimizer = tf.train.RMSPropOptimizer(self.lr, decay=0.9, 
								momentum=0.0, name='RMSProp')
		elif self.optim == 'Adam':
			self.optimizer = tf.train.AdamOptimizer(self.lr, name='Adam')

	def convert_to_space(self, text, vis):
		""" Convert the raw features into the new space, only useful
			when the self.mode == 'end' or 'double'.
			state is trainable True or False.
		"""
		if self.mode == 'end':
			state = True
		else:
			state = False
			
		with tf.name_scope("text_encode"):
			text = tf.layers.dense(text, trainable=state,
				activation=self.activation_func, reuse=tf.AUTO_REUSE,
				units=self.embed_size, name='text_forward_layer1')
			text = tf.layers.dropout(text, rate=self.dropout)
			text_forward = tf.layers.dense(text, trainable=state,
				activation=self.activation_func, reuse=tf.AUTO_REUSE,
				units=self.embed_size, name='text_forward_layer2')

		with tf.name_scope("vis_encode"):
			vis = tf.layers.dense(vis, trainable=state,
				activation=self.activation_func, reuse=tf.AUTO_REUSE,
				units=self.embed_size, name='vis_forward_layer1')
			vis = tf.layers.dropout(vis, rate=self.dropout)
			vis_forward = tf.layers.dense(vis, trainable=state,
				activation=self.activation_func, reuse=tf.AUTO_REUSE,
				units=self.embed_size, name='vis_forward_layer2')

		if self.mode == 'text':
			return text_forward
		elif self.mode == 'vis':
			return vis_forward
		else:
			return text_forward, vis_forward

	def item_FC(self, item):
		""" item fully connected layers. """
		if self.mode in ['end', 'double']:
			item = tf.layers.dense(item, 
					activation=self.activation_func,	
					units=self.embed_size, 
					reuse=tf.AUTO_REUSE, 
					name='item_FC')
		else:
			item = tf.layers.dense(item, 
					activation=self.activation_func,	
					units=self.embed_size, 
					reuse=tf.AUTO_REUSE, 
					name='item_FC1')
			item = tf.layers.dense(item, 
					activation=self.activation_func,	
					units=self.embed_size, 
					reuse=tf.AUTO_REUSE, 
					name='item_FC2')

		return item

	def triplet_loss(self, anchor, pos, neg, margin=1.0):
		neg = tf.transpose(neg, perm=[1, 0, 2])
		dis_pos = tf.sqrt(tf.reduce_sum(tf.square(anchor-pos), -1) + 1e-6)
		dis_neg = tf.sqrt(tf.reduce_sum(tf.square(anchor-neg), -1) + 1e-6)

		loss = tf.maximum(0.0, margin + dis_pos - dis_neg)
		loss = tf.reduce_mean(loss)

		return loss

	def create_model(self):
		""" Create model from scratch. """
		self.user_onehot = tf.one_hot(self.user, self.user_size)
		self.user_embed = tf.layers.dense(inputs=self.user_onehot,
									units=self.embed_size,
									activation=self.activation_func,
									kernel_regularizer=self.regularizer,
									name='user_embed')
		self.query_embed = tf.layers.dense(inputs=self.query, 
									activation=self.activation_func,
									units=self.embed_size,
									kernel_regularizer=self.regularizer,
									name='query_embed')

		self.user_trans = tf.layers.dense(inputs=self.user_embed, 
									activation=self.activation_func,
									units=self.embed_size,
									reuse=tf.AUTO_REUSE,
									name='translate')
		self.query_trans = tf.layers.dense(inputs=self.query_embed, 
									activation=self.activation_func,
									units=self.embed_size,
									reuse=tf.AUTO_REUSE,
									name='translate')
		self.item_predict = self.user_trans + self.query_embed

		if self.mode == 'vis':
			self.pos_vis_conv = self.convert_to_space(self.pos_text, self.pos_vis)
			self.neg_vis_conv = self.convert_to_space(self.neg_text, self.neg_vis)
			self.pos_cat = self.pos_vis_conv
			self.neg_cat = self.neg_vis_conv
		elif self.mode == 'text':
			self.pos_text_conv = self.convert_to_space(self.pos_text, self.pos_vis)
			self.neg_text_conv = self.convert_to_space(self.neg_text, self.neg_vis)
			self.pos_cat = self.pos_text_conv
			self.neg_cat = self.neg_text_conv
		else:
			self.pos_text_conv, self.pos_vis_conv = self.convert_to_space(
									self.pos_text, self.pos_vis)
			self.pos_cat = tf.concat([self.pos_text_conv, self.pos_vis_conv], -1)

			self.neg_text_conv, self.neg_vis_conv = self.convert_to_space(
									self.neg_text, self.neg_vis)
			self.neg_cat = tf.concat([self.neg_text_conv, self.neg_vis_conv], -1)

		self.pos_item = self.item_FC(self.pos_cat)
		self.pos_item_trans = tf.layers.dense(inputs=self.pos_item, 
									activation=self.activation_func,
									units=self.embed_size,
									reuse=tf.AUTO_REUSE,
									name='translate')

		self.neg_item = self.item_FC(self.neg_cat)
		self.neg_item_trans = tf.layers.dense(inputs=self.neg_item, 
									activation=self.activation_func,
									units=self.embed_size,
									reuse=tf.AUTO_REUSE,
									name='translate')

	def loss_func(self):
		reg = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
		reg_loss = tf.contrib.layers.apply_regularization(
									self.regularizer, reg)
		self.final_loss = self.triplet_loss(self.item_predict, self.pos_item_trans, 
						self.neg_item_trans) + reg_loss

	def optimization(self):
		with tf.name_scope("optimization"):
			self.optim = self.optimizer.minimize(self.final_loss)

	def eval(self):
		scores = tf.norm(self.item_predict - self.all_items, axis=-1)
		_, self.indices = tf.nn.top_k(scores, self.item_size)

	def summary(self):
		""" Create summaries to write on tensorboard. """
		self.writer = tf.summary.FileWriter(
					'./graphs/TranSearch', tf.get_default_graph())
		with tf.name_scope("summaries"):
			tf.summary.scalar('loss', self.final_loss)
			self.summary_op = tf.summary.merge_all()

	def build(self):
		self.inference()
		self.create_model()
		self.loss_func()
		self.optimization()
		self.eval()
		self.summary()
		self.saver = tf.train.Saver(tf.global_variables())
		reuse_vars_text_layer1 = tf.get_collection(
				tf.GraphKeys.GLOBAL_VARIABLES, scope='text_forward_layer1')
		reuse_vars_text_layer2 = tf.get_collection(
				tf.GraphKeys.GLOBAL_VARIABLES, scope='text_forward_layer2')
		reuse_vars_vis_layer1 = tf.get_collection(
				tf.GraphKeys.GLOBAL_VARIABLES, scope='vis_forward_layer1')
		reuse_vars_vis_layer2 = tf.get_collection(
				tf.GraphKeys.GLOBAL_VARIABLES, scope='vis_forward_layer2')
		reuse_vars = (reuse_vars_text_layer1 + reuse_vars_text_layer2 + 
						reuse_vars_vis_layer1 + reuse_vars_vis_layer2)
		reuse_vars_dict = dict([(var.op.name, var) for var in reuse_vars])
		self.pre_train = tf.train.Saver(reuse_vars_dict)

	def step(self, sess, sample, all_items_embed, step):
		input_feed = {}
		if self.is_training:
			input_feed[self.user.name] = sample[0]
			input_feed[self.query.name] = sample[1]
			input_feed[self.pos_text.name] = sample[2]
			input_feed[self.pos_vis.name] = sample[3]
			input_feed[self.neg_text.name] = sample[4]
			input_feed[self.neg_vis.name] = sample[5]

			output_feed = [self.optim, self.summary_op]

			outputs = sess.run(output_feed, input_feed)

			self.writer.add_summary(outputs[-1], global_step=step)
		else:
			if self.test_first:
				input_feed[self.pos_text.name] = sample[0]
				input_feed[self.pos_vis.name] = sample[1]

				output_feed = [self.pos_item_trans]

				item_embed = sess.run(output_feed, input_feed)

				return item_embed 

			else:
				input_feed[self.user.name] = sample[0]
				input_feed[self.query.name] = sample[1]
				input_feed[self.all_items.name] = all_items_embed

				output_feed = [self.indices]

				item_indices = sess.run(output_feed, input_feed)

				return item_indices
