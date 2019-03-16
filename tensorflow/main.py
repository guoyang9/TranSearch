from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys, time

import numpy as np 
import tensorflow as tf 
import data_input
from TranSearch import TranSearch
import metrics


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('visual_size', 4096, 
			'size of image feature.')
tf.app.flags.DEFINE_integer('text_size', 512, 
			'size of textual feature.')
tf.app.flags.DEFINE_integer('batch_size', 256, 
			'size of mini-batch.')
tf.app.flags.DEFINE_integer('negative_num', 5, 
			'number of negative samples.')
tf.app.flags.DEFINE_integer('embed_size', 32, 
			'the size for embedding user and item.')
tf.app.flags.DEFINE_integer('topK', 20, 
			'truncated top items.')
tf.app.flags.DEFINE_integer('epochs', 20, 
			'the number of epochs.')
tf.app.flags.DEFINE_string('dataset', 'MenClothing', 
			'the pre-trained dataset.')
tf.app.flags.DEFINE_string('model_dir', './TranSearch/', 
			'the dir for saving model.')
tf.app.flags.DEFINE_string('mode', 'end', 
			'could be "end", "vis", "text", "double".')
tf.app.flags.DEFINE_string('optim', 'Adam', 
			'the optimization method.')
tf.app.flags.DEFINE_string('activation', 'ELU', 
			'the activation function.')
tf.app.flags.DEFINE_string('gpu', '0', 
			'the gpu card number.')
tf.app.flags.DEFINE_float('lr', 0.001, 
			'learning rate.')
tf.app.flags.DEFINE_float('dropout', 0.5, 
			'dropout rate.')
tf.app.flags.DEFINE_float('l2_rate', 0.0001, 
			'regularize rate.')

opt_gpu = FLAGS.gpu
os.environ["CUDA_VISIBLE_DEVICES"] = opt_gpu
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

def main(argv=None):
	train_data = data_input.TranSearchData(
			FLAGS.dataset, 'train.csv', FLAGS.negative_num, True)
	test_data = data_input.TranSearchData(
			FLAGS.dataset, 'test.csv', FLAGS.negative_num, False)

	train_dataset = tf.data.Dataset.from_generator(train_data.get_instance,
				output_types=(tf.int32, tf.float32, tf.float32, 
							tf.float32, tf.float32, tf.float32),
				output_shapes=(tf.TensorShape([]), tf.TensorShape([512]),
							tf.TensorShape([512]), tf.TensorShape([4096]),
							tf.TensorShape([FLAGS.negative_num, 512]), 
							tf.TensorShape([FLAGS.negative_num, 4096])))

	train_dataset = train_dataset.shuffle(100000).batch(FLAGS.batch_size)
	train_iter = train_dataset.make_initializable_iterator()
	
	all_items_idx = []
	for _, _, item in train_data.get_all_test():
		all_items_idx.append(item)

	user_size = len(train_data.data.userID.unique())
	item_size = len(all_items_idx)
	all_items_idx = np.array(all_items_idx)
	user_bought = train_data.user_bought
	with tf.Session(config=config) as sess:
		train(sess, train_iter, item_size, user_size, 
						test_data, all_items_idx, user_bought)

def train(sess, train_iter, item_size, user_size, 
						test_data, all_items_idx, user_bought):

	############################### CREATE MODEL #############################
	
	model = TranSearch(FLAGS.visual_size, FLAGS.text_size, FLAGS.embed_size,
						item_size, user_size, FLAGS.mode, FLAGS.lr, 
						FLAGS.l2_rate, FLAGS.optim, FLAGS.activation, 
						False, FLAGS.dropout, FLAGS.negative_num, is_training=True)
	model.build()

	ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
	if ckpt:
		print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
		model.saver.restore(sess, ckpt.model_checkpoint_path)
	else:
		print("Creating model with fresh parameters.")
		sess.run(tf.global_variables_initializer())

		if FLAGS.mode != 'end':
			model.pre_train.restore(sess, './Variables/pre_train.ckpt')

	############################### Training ####################################

	count = 0
	for epoch in range(FLAGS.epochs):
		model.is_training = True
		start_time = time.time()
		sess.run(train_iter.initializer)
		train_next = train_iter.get_next()

		try:
			while True:
				sample = sess.run(train_next)
				model.step(sess, sample, None, count)
				count += 1
		except tf.errors.OutOfRangeError:
			test(model, sess, test_data, all_items_idx, user_bought)
			print("Epoch %d " %epoch + "Took: " + time.strftime("%H: %M: %S", 
							time.gmtime(time.time() - start_time)))

	################################## SAVE MODEL ################################

	# checkpoint_path = os.path.join(FLAGS.model_dir, "TranSearch.ckpt")
	# model.saver.save(sess, checkpoint_path)
	
def test(model, sess, test_data, all_items_idx, user_bought):
	model.is_training = False
	model.test_first = True
	all_items_embed = []
	HR, MRR, NDCG = [], [], []

	########################## GET ALL ITEM EMBEDDING ONCE ######################

	for sample in test_data.get_all_test():
		item_embed = model.step(sess, sample, None, None)
		all_items_embed.append(item_embed[0][0])

	model.test_first = False
	all_items_embed = np.array(all_items_embed)

	########################## TEST FOR EACH USER QUERY PAIR #####################

	for sample in test_data.get_instance():
		item_indices = model.step(sess, sample, all_items_embed, None)[0]
		itemID = sample[3]
		reviewerID = sample[4]

		ranking_list = all_items_idx[item_indices].tolist()

		top_idx = []
		u_bought = user_bought[reviewerID]
		while len(top_idx) < FLAGS.topK:
			candidate_item = ranking_list.pop()
			if candidate_item not in u_bought or candidate_item == itemID:
				top_idx.append(candidate_item)
		top_idx = np.array(top_idx)

		HR.append(metrics.hit(itemID, top_idx))
		MRR.append(metrics.mrr(itemID, top_idx))
		NDCG.append(metrics.ndcg(itemID, top_idx))

	hr = np.array(HR).mean()
	mrr = np.array(MRR).mean()
	ndcg = np.array(NDCG).mean()
	print("HR is %.3f, MRR is %.3f, NDCG is %.3f" %(hr, mrr, ndcg))


if __name__ == '__main__':
	tf.app.run()
