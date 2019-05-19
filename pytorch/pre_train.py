from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
import time
import json
import argparse
sys.path.append(os.getcwd())

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import config
from data_input import PretrainData


class PrepareFeatures(nn.Module):
	def __init__(self, visual_size, text_size, 
						embed_size, dropout, is_training):
		super(PrepareFeatures, self).__init__()
		"""
			For both visual and textual parts, we use several fully 
			connnected layers, visual_size is 4096, textual_size is 512. 
		"""
		self.is_training = is_training

		# visual fully connected layers
		self.visual_FC_forward = nn.Sequential(
			nn.Linear(visual_size, embed_size),
			nn.ELU(),
			nn.Dropout(p=dropout),			
			nn.Linear(embed_size, embed_size),
			nn.ELU())
		self.visual_FC_backward = nn.Sequential(
			nn.Linear(embed_size, embed_size),
			nn.ELU(),
			nn.Dropout(p=dropout),
			nn.Linear(embed_size, visual_size),
			nn.ELU())

		# textual fully connected layers
		self.textual_FC_forward = nn.Sequential(
			nn.Linear(text_size, embed_size),
			nn.ELU(),
			nn.Dropout(p=dropout),			
			nn.Linear(embed_size, embed_size),
			nn.ELU())
		self.textual_FC_backward = nn.Sequential(
			nn.Linear(embed_size, embed_size),
			nn.ELU(),
			nn.Dropout(p=dropout),			
			nn.Linear(embed_size, text_size),
			nn.ELU())

		for m in self.modules():
			if isinstance(m, nn.Linear):
				nn.init.xavier_uniform_(m.weight)
				if m.bias is not None:
					m.bias.data.zero_()
		
	def forward(self, anchor_text, anchor_vis, pos_text, 
									pos_vis, neg_text, neg_vis):
		if self.is_training:
			av_encode = self.visual_FC_forward(anchor_vis)
			at_encode = self.textual_FC_forward(anchor_text)
			pv_encode = self.visual_FC_forward(pos_vis)
			pt_encode = self.textual_FC_forward(pos_text)
			nv_encode = self.visual_FC_forward(neg_vis)
			nt_encode = self.textual_FC_forward(neg_text)

			av_decode = self.visual_FC_backward(av_encode)
			at_decode = self.textual_FC_backward(at_encode)
			pv_decode = self.visual_FC_backward(pv_encode)
			pt_decode = self.textual_FC_backward(pt_encode)
			nv_decode = self.visual_FC_backward(nv_encode)
			nt_decode = self.textual_FC_backward(nt_encode)

			return ((av_encode, pv_encode, nv_encode),
					(at_encode, pt_encode, nt_encode),
					(av_decode, pv_decode, nv_decode),
					(at_decode, pt_decode, nt_decode))
		else:
			av_encode = self.visual_FC_forward(anchor_vis)
			at_encode = self.textual_FC_forward(anchor_text)
			return av_encode, at_encode


def main():
	torch.multiprocessing.set_sharing_strategy('file_system')

	parser = argparse.ArgumentParser()
	parser.add_argument("--embed_size", 
		type=int,
		default=32, 
		help="the final embedding size")
	parser.add_argument("--lr", 
		type=float,
		default=0.001, 
		help="the learning rate for optimization")
	parser.add_argument("--dropout", 
		type=float,
		default=0.5, 
		help="the dropout rate")
	parser.add_argument("--neg_number", 
		type=int,
		default=20, 
		help="negative numbers for training the triplet model")
	parser.add_argument("--batch_size", 
		type=int,
		default=512, 
		help="batch size for training")
	parser.add_argument("--epochs", 
		type=int,
		default=20, 
		help="number of epochs")
	parser.add_argument("--gpu", 
		type=str,
		default='0', 
		help="choose the gpu card number")
	FLAGS = parser.parse_args()


	writer = SummaryWriter() # for visualization

	opt_gpu = FLAGS.gpu
	os.environ["CUDA_VISIBLE_DEVICES"] = opt_gpu
	cudnn.benchmark = True

	#Create model.
	model = PrepareFeatures(
		visual_size=config.visual_size, 
		text_size=config.textual_size, 
		embed_size=FLAGS.embed_size, 
		dropout=FLAGS.dropout, is_training=True)
	model.cuda()
	optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.lr)

	# Prepare input data.
	data = PretrainData(FLAGS.neg_number)
	print("Sampling negative items for each positive pairs...")
	print("Start training......")

	for epoch in range(FLAGS.epochs):
		model.is_training = True
		model.train() 
		start_time = time.time()

		data.sample_neg()
		dataloader = DataLoader(data, 
			batch_size=FLAGS.batch_size, shuffle=False, num_workers=4)

		for idx, batch_data in enumerate(dataloader):
			at = batch_data['anchor_text'].cuda()
			av = batch_data['anchor_vis'].cuda()
			pt = batch_data['pos_text'].cuda()
			pv = batch_data['pos_vis'].cuda()
			nt = batch_data['neg_text'].cuda()
			nv = batch_data['neg_vis'].cuda()

			model.zero_grad()
			((av_encode, pv_encode, nv_encode), 
				(at_encode, pt_encode, nt_encode),
				(av_decode, pv_decode, nv_decode), 
				(at_decode, pt_decode, nt_decode)) = model(at, av, pt, pv, nt, nv)

			# triplet loss for the two distinctive features
			triplet_loss = F.triplet_margin_loss(av_encode, pv_encode, nv_encode
					) + F.triplet_margin_loss(at_encode, pt_encode, nt_encode) 
			reconstruc_loss = 0.01*(
				   (F.mse_loss(av, av_decode) + F.mse_loss(at, at_decode) +
				   F.mse_loss(pv, pv_decode) + F.mse_loss(pt, pt_decode) +
				   F.mse_loss(nv, nv_decode)) + F.mse_loss(nt, nt_decode))
			loss = triplet_loss + reconstruc_loss

			loss.backward()
			optimizer.step()

			writer.add_scalar('data/triplet_loss', triplet_loss.data.item(),
										epoch*len(dataloader)+idx)
			writer.add_scalar('data/reconstruc_loss', reconstruc_loss.data.item(),
										epoch*len(dataloader)+idx)

		elapsed_time = time.time() - start_time
		print("Epoch: {:d}\t".format(epoch) + "Epoch time: " +
			time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))

	model.eval()
	model.is_training = False

	# save the pre-trained weights
	if not os.path.exists(config.weights_path):
		os.makedirs(config.weights_path)
	torch.save(model.visual_FC_forward, config.image_weights_path)
	torch.save(model.textual_FC_forward, config.text_weights_path)


if __name__ == "__main__":
	main()
