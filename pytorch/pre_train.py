from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, argparse, time, json, sys

import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data_input import pretrainData


class prepareFeatures(nn.Module):
	def __init__(self, visual_size, text_size, 
						embed_size, dropout, is_training):
		super(prepareFeatures, self).__init__()
		"""
			For both visual and textual parts, we use 
			several fully connnected layers,
			visual_size is 4096, textual_size is 512.
		"""
		self.is_training = is_training

		#Custom weights initialization.
		def init_weights(m):
			if type(m) == nn.Linear:
				nn.init.xavier_uniform_(m.weight)
				nn.init.constant_(m.bias, 0)

		#Visual fully connected layers.
		self.visual_FC_forward = nn.Sequential(
			nn.Linear(visual_size, embed_size),
			nn.ELU(),
			nn.Dropout(p=dropout),			
			nn.Linear(embed_size, embed_size),
			nn.ELU()
		)
		self.visual_FC_backward = nn.Sequential(
			nn.Linear(embed_size, embed_size),
			nn.ELU(),
			nn.Dropout(p=dropout),
			nn.Linear(embed_size, visual_size),
			nn.ELU()
		)

		#Textual fully connected layers.
		self.textual_FC_forward = nn.Sequential(
			nn.Linear(text_size, embed_size),
			nn.ELU(),
			nn.Dropout(p=dropout),			
			nn.Linear(embed_size, embed_size),
			nn.ELU()
		)
		self.textual_FC_backward = nn.Sequential(
			nn.Linear(embed_size, embed_size),
			nn.ELU(),
			nn.Dropout(p=dropout),			
			nn.Linear(embed_size, text_size),
			nn.ELU()
		)

		self.visual_FC_forward.apply(init_weights)
		self.visual_FC_backward.apply(init_weights)
		self.textual_FC_forward.apply(init_weights)
		self.textual_FC_backward.apply(init_weights)

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

	parser.add_argument("--path", default='../processed/', type=str,
				help="the data path.")
	parser.add_argument("--dataset", default='MenClothing', type=str,
				help="choose dataset to process.")
	parser.add_argument("--embed_size", default=32, type=int,
				help="the final embedding size.")
	parser.add_argument("--lr", default=0.001, type=float,
				help="the learning rate for optimization method.")
	parser.add_argument("--dropout", default=0.5, type=float,
				help="the dropout rate.")
	parser.add_argument("--neg_number", default=20, type=int,
				help="negative numbers for training the triplet model.")
	parser.add_argument("--batch_size", default=512, type=int,
				help="batch size for training.")
	parser.add_argument("--epochs", default=20, type=int,
				help="number of epochs.")
	parser.add_argument("--gpu", default='0', type=str,
				help="choose the gpu card number.")

	FLAGS = parser.parse_args()
	ROOT_DIR = '../processed/'

	writer = SummaryWriter() #For visualization

	opt_gpu = FLAGS.gpu
	os.environ["CUDA_VISIBLE_DEVICES"] = opt_gpu

	#Create model.
	model = prepareFeatures(visual_size=4096, text_size=512, 
							embed_size=FLAGS.embed_size, 
							dropout=FLAGS.dropout, is_training=True)
	model.cuda()
	# optimizer = torch.optim.SGD(model.parameters(), 
	#             momentum=0.9, lr=FLAGS.lr, weight_decay=0.0001)
	optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.lr)
	# scheduler = ReduceLROnPlateau(optimizer, min_lr=1e-08, patience=10)

	# Prepare input data.
	data = pretrainData(FLAGS.dataset)
	print("Sampling negative items for each positive pairs......\n")
	print("Start training......")

	for epoch in range(FLAGS.epochs):
		model.is_training = True
		model.train() #Enable dropout.
		start_time = time.time()

		data.sample_neg(FLAGS.neg_number)
		dataloader = DataLoader(data, batch_size=FLAGS.batch_size,
					shuffle=False, num_workers=4)

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

			# Triplet loss for the two distinctive features.
			triplet_loss = F.triplet_margin_loss(av_encode, pv_encode, nv_encode
					) + F.triplet_margin_loss(at_encode, pt_encode, nt_encode) 
			reconstruc_loss = 0.01*(
				   (F.mse_loss(av, av_decode) + F.mse_loss(at, at_decode) +
				   F.mse_loss(pv, pv_decode) + F.mse_loss(pt, pt_decode) +
				   F.mse_loss(nv, nv_decode)) + F.mse_loss(nt, nt_decode))
			loss = triplet_loss + reconstruc_loss

			loss.backward()
			optimizer.step()
			# scheduler.step(loss.data[0])

			writer.add_scalar('data/triplet_loss', triplet_loss.data.item(),
			                            epoch*len(dataloader)+idx)
			writer.add_scalar('data/reconstruc_loss', reconstruc_loss.data.item(),
			                            epoch*len(dataloader)+idx)

		elapsed_time = time.time() - start_time
		print("Epoch: %d\t" %(epoch) + "Epoch time: " +
			time.strftime("%H: %M: %S", time.gmtime(elapsed_time))+'\n')

	model.eval()
	model.is_training = False

	# Save the pre-trained weights
	torch.save(model.visual_FC_forward, './Variable/visual_FC.pt')
	torch.save(model.textual_FC_forward, './Variable/textual_FC.pt')


if __name__ == "__main__":
	main()
