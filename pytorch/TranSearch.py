from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, argparse, time, sys

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data_input import TranSearchData
import evaluate

ROOT_DIR = '../processed/'

class TranSearch(nn.Module):
	def __init__(self, visual_FC, textual_FC, visual_size, 
						text_size, embed_size, 
						user_size, mode, dropout, is_training):
		super(TranSearch, self).__init__()
		""" 
		Important Args:
		visual_size: for end_to_end is 4096, for others is not.
		text_size: for end_to_end is 512, for others is not.
		mode: could be 'end', vis', 'text', 'double'.
		"""
		self.visual_size = visual_size
		self.text_size = text_size
		self.embed_size = embed_size
		self.user_size = user_size
		self.mode = mode
		self.is_training = is_training

		#Custom weights initialization.
		def init_weights(m):
			if type(m) == nn.Linear:
				nn.init.xavier_uniform_(m.weight)
				m.bias.data.fill_(0)

		if self.mode == 'end':
			# Visual fully connected layers.
			self.visual_FC = nn.Sequential(
				nn.Linear(visual_size, embed_size),
				nn.ELU(),
				nn.Dropout(p=dropout),
				nn.Linear(embed_size, embed_size),
				nn.ELU()
			)

			# Textual fully connected layers.
			self.textual_FC = nn.Sequential(
				nn.Linear(text_size, embed_size),
				nn.ELU(),
				nn.Dropout(p=dropout),
				nn.Linear(embed_size, embed_size),
				nn.ELU()
			)
			self.visual_FC.apply(init_weights)
			self.textual_FC.apply(init_weights)
		else:
			self.visual_FC = visual_FC
			self.textual_FC = textual_FC

		# User and query embedding.
		self.user_embed = nn.Sequential(
			nn.Linear(self.user_size, embed_size),
			nn.ELU()
		)

		self.query_embed = nn.Sequential(
			nn.Linear(text_size, embed_size),
			nn.ELU()
		)
		self.user_embed.apply(init_weights)
		self.query_embed.apply(init_weights)

		# For embed user and item in the same space
		self.translation = nn.Sequential(
			nn.Linear(embed_size, embed_size),
			nn.ELU()
		)
		self.translation.apply(init_weights)

		# Item fully connected layers.
		if self.mode in ['end', 'double']:
			self.item_FC = nn.Sequential(
				nn.Linear(2*embed_size, embed_size),
				# nn.ELU(),
				# nn.Dropout(p=dropout),
				# nn.Linear(embed_size, embed_size),
				nn.ELU()
			)
		else:
			self.item_FC = nn.Sequential(
				nn.Linear(embed_size, embed_size),
				nn.ELU(),
				nn.Linear(embed_size, embed_size),
				nn.ELU()
			)
		self.item_FC.apply(init_weights)

	def convert_onehot(self, feature):
		"""Convert user to one-hot format"""
		batch_size = feature.shape[0]
		feature = feature.view(-1, 1)
		f_onehot = torch.cuda.FloatTensor(batch_size, self.user_size)
		f_onehot.zero_()
		f_onehot.scatter_(1, feature.data, 1)

		return f_onehot

	def forward(self, user, query, pos_vis, pos_text, 
						neg_vis, neg_text, test_first):
		if not test_first:
			user = self.convert_onehot(user)
			user = self.user_embed(user)
			user = self.translation(user)
			query = self.translation(self.query_embed(query))
			item_predict = user + query

		if self.is_training or test_first:
			# Postive features attention and concatenation.
			if self.mode == 'vis':
				pos_vis = self.visual_FC(pos_vis)
				pos_concat = pos_vis
			elif self.mode == 'text':
				pos_text = self.textual_FC(pos_text)
				pos_concat = pos_text
			else:
				pos_vis = self.visual_FC(pos_vis)
				pos_text = self.textual_FC(pos_text)
				pos_concat = torch.cat((pos_vis, pos_text), dim=-1)
				# pos_concat = pos_vis * pos_text
			pos_item = self.item_FC(pos_concat)
			pos_item = self.translation(pos_item)

		if self.is_training:
			# Negative features attention and concatenation.
			if self.mode == 'vis':
				neg_vis = self.visual_FC(neg_vis)
				neg_concat = neg_vis
			elif self.mode == 'text':
				neg_text = self.textual_FC(neg_text)
				neg_concat = neg_text
			else:
				neg_vis = self.visual_FC(neg_vis)
				neg_text = self.textual_FC(neg_text)
				neg_concat = torch.cat((neg_vis, neg_text), dim=-1)
				# neg_concat = neg_vis * neg_text
			neg_items = self.item_FC(neg_concat)
			neg_items =self.translation(neg_items)

			return item_predict, pos_item, neg_items

		else:
			if test_first:
				return pos_item
			else:
				return item_predict

def TripletLoss(anchor, positive, negatives):
	""" 
	We found that add all the negative ones together can 
	yeild better performance.
	"""
    batch_size = negatives.shape[0]
    neg_num = negatives.shape[1]
    embed_size = negatives.shape[2]
    negatives = negatives.view(neg_num, batch_size, embed_size)

    losses = 0
    for idx, negative in enumerate(negatives):
        losses += torch.mean(
        	F.triplet_margin_loss(anchor, positive, negative))

    return losses/(idx+1)

def main():
	parser = argparse.ArgumentParser()

	parser.add_argument("--dataset", default='MenClothing', type=str,
				help="choose dataset to process.")
	parser.add_argument("--embed_size", default=32, type=int,
				help="the final embedding size.")
	parser.add_argument("--lr", default=0.001, type=float,
				help="the learning rate for optimization method.")
	parser.add_argument("--dropout", default=0.5, type=float,
				help="the dropout rate.")
	parser.add_argument("--neg_number", default=5, type=int,
				help="negative numbers for training the triplet model.")
	parser.add_argument("--batch_size", default=512, type=int,
				help="batch size for training.")
	parser.add_argument("--top_k", default=20, type=int,
				help="topk rank items for evaluating.")
	parser.add_argument("--is_output", default=False, type=bool,
				help="output the result for rank test.")
	parser.add_argument("--mode", default='double', type=str,
				help="the model mode.")
	parser.add_argument("--gpu", default='0', type=str,
				help="choose the gpu card number.")

	FLAGS = parser.parse_args()

	writer = SummaryWriter() #For visualization

	opt_gpu = FLAGS.gpu
	os.environ["CUDA_VISIBLE_DEVICES"] = opt_gpu

	############################# PREPARE DATASET ##########################

	data_train = TranSearchData(
				FLAGS.dataset, 'train.csv', is_training=True)
	data_test  = TranSearchData(
				FLAGS.dataset, 'test.csv', is_training=False)
	print("Sampling negative items for each positive pairs......\n")
	data_train.sample_neg(FLAGS.neg_number)
	dataloader_train = DataLoader(data_train, 
			batch_size=FLAGS.batch_size, shuffle=True, num_workers=4)
	data_test.sample_neg(0)
	dataloader_test = DataLoader(data_test, shuffle=False, batch_size=1)

	####################### LOAD PRE-TRAIN WEIGHTS ##########################

	visual_FC = torch.load('./Variable/visual_FC.pt')
	#First remove the dropout layer.
	modules = list(visual_FC.children())[:2] + list(visual_FC.children())[3:]
	visual_FC = nn.Sequential(*modules)
	visual_FC.requires_grad=False
	textual_FC = torch.load('./Variable/textual_FC.pt')
	modules = list(textual_FC.children())[:2] + list(textual_FC.children())[3:]
	textual_FC = nn.Sequential(*modules)
	textual_FC.requires_grad=False

	############################## CREATE MODEL ###########################

	full_data = pd.read_csv(os.path.join(ROOT_DIR, 
					FLAGS.dataset, 'full.csv'), usecols=['userID'])
	user_size = len(full_data.userID.unique())

	# Create model.
	model = TranSearch(visual_FC, textual_FC, 4096, 512, FLAGS.embed_size,
					user_size, FLAGS.mode, FLAGS.dropout, is_training=True)
	model.cuda()
	# optimizer = torch.optim.SGD(
	# 					 model.parameters(), momentum=0.9, lr=0.01)
	#scheduler = ReduceLROnPlateau(optimizer, min_lr=1e-08, patience=30)
	optimizer = torch.optim.Adam(
				model.parameters(), lr=FLAGS.lr, weight_decay=0.0001)

	print("Start training......\n")
	for epoch in range(20):
		model.is_training = True
		model.train() 
		start_time = time.time()

		for idx, batch_data in enumerate(dataloader_train):
			user = batch_data['userID'].cuda()
			query = batch_data['query'].cuda()
			pos_vis = batch_data['pos_vis'].cuda()
			pos_text = batch_data['pos_text'].cuda()
			neg_vis = batch_data['neg_vis'].cuda()
			neg_text = batch_data['neg_text'].cuda()

			model.zero_grad()

			item_predict, pos_item, neg_items = model(user, query,
						pos_vis, pos_text, neg_vis, neg_text, False)
			loss = TripletLoss(item_predict, pos_item, neg_items)

			loss.backward()
			optimizer.step()
			# scheduler.step(loss.data[0])

			writer.add_scalar('data/endtoend_loss', loss.data.item(),
			                                epoch*len(dataloader_train)+idx)

		print("Epoch %d training is done!\n" %epoch)

		# Start testing
		model.eval() 
		model.is_training = False
		Mrr, Hr, Ndcg = evaluate.metrics(model, data_test,
					dataloader_test, FLAGS.top_k, FLAGS.is_output, epoch)
			
		elapsed_time = time.time() - start_time
		print("Epoch: %d\t" %epoch + "Epoch time: " + time.strftime(
							"%H: %M: %S", time.gmtime(elapsed_time)))
		print("Mrr is %.3f.\nHit ratio is %.3f.\nNdcg is %.3f.\n" %(
																Mrr, Hr, Ndcg))


if __name__ == "__main__":
	main()
