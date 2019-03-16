from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json, os, sys

import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec
from torch.utils.data import Dataset

ROOT_DIR = '../processed/'


class pretrainData(Dataset):
	def __init__(self, dataset):
		""" For pretraining the image and review features."""
		self.data_path = os.path.join(ROOT_DIR, dataset)
		self.asin_dict = json.load(open(
			os.path.join(self.data_path, 'asin_sample.json'), 'r'))

		self.all_items = set(self.asin_dict.keys())
		# Prepare textual feture data.
		self.doc2vec_model = Doc2Vec.load(
							os.path.join(self.data_path, 'doc2vecFile'))
		self.text_vec = {}
		for asin in self.asin_dict:
			self.text_vec[asin] = self.doc2vec_model.docvecs[asin]

		# Prepare visual feature data.
		self.vis_vec = np.load(os.path.join(
					ROOT_DIR, dataset, 'image_feature.npy')).item()

	def sample_neg(self, neg_number):
		""" Sample the anchor, positive, negative tuples."""
		self.features = []
		for asin in self.asin_dict:
			pos_items = self.asin_dict[asin]['positive']
			if not len(pos_items) == 0:
				for pos in pos_items:
					neg = np.random.choice(list(
								self.all_items - set(pos_items)),
								neg_number, replace=False)
					for n in neg:
						self.features.append((asin, pos, n))

	def __len__(self):
		""" For each anchor item, sample neg_number items."""
		return len(self.features)

	def test(self):
		for asin in self.asin_dict:
			anchor_text = self.text_vec[asin]
			anchor_vis = self.vis_vec[asin]

			yield (anchor_text, anchor_vis, asin)

	def __getitem__(self, idx):
		feature_idx = self.features[idx]
		anchor_item = feature_idx[0]
		pos_item = feature_idx[1]
		neg_item = feature_idx[2]

		anchor_vis = self.vis_vec[anchor_item]
		anchor_text = self.text_vec[anchor_item]
		pos_vis = self.vis_vec[pos_item]
		pos_text = self.text_vec[pos_item]
		neg_vis = self.vis_vec[neg_item]
		neg_text = self.text_vec[neg_item]

		sample = {'anchor_text': anchor_text, 
				'anchor_vis': anchor_vis,
				'pos_text': pos_text, 'pos_vis': pos_vis,
				'neg_text': neg_text, 'neg_vis': neg_vis}

		return sample


class TranSearchData(pretrainData):
	def __init__(self, dataset, datafile, is_training):
		""" Without pre-train, input the raw data."""
		super().__init__(dataset)
		self.is_training = is_training
		self.data = pd.read_csv(
						os.path.join(self.data_path, datafile))
		self.query_dict = json.load(open(
				os.path.join(self.data_path, 'queryFile.json'), 'r'))
		self.user_bought = json.load(open(
				os.path.join(self.data_path, 'user_bought.json'), 'r'))
		self.items = list(self.asin_dict.keys())

	def sample_neg(self, neg_number):
		""" Take the also_view or buy_after_viewing 
			as negative samples.
		"""
		self.features = []
		for i in range(len(self.data)):
			query_vec = self.doc2vec_model.docvecs[
								self.query_dict[self.data['query_'][i]]]
			if self.is_training:
				# We tend to sample negative ones from the 
				# also_view and buy_after_viewing items, if don't 
				# have enough, we then randomly sample negative 
				# ones(which stored in 'negative').
				asin = self.data['asin'][i]
				sample = self.asin_dict[asin]
				all_sample = sample['positive'] + sample['negative']
				negs = np.random.choice(
					all_sample, neg_number, replace=False, p=sample['prob'])
				# negs = np.random.choice(
				# 				self.items, neg_number, replace=False)
				# while asin in negs:
				# 	negs = np.random.choice(
				# 				self.items, neg_number, replace=False)
			
				self.features.append(((
						self.data['userID'][i], query_vec), (asin, negs)))

			else:
				self.features.append(((self.data['userID'][i], query_vec),
					(self.data['reviewerID'][i], self.data['asin'][i]),
					self.data['query_'][i]))

	def get_all_test(self):
		for asin in self.asin_dict:
			sample_vis = self.vis_vec[asin]
			sample_text = self.text_vec[asin]

			yield sample_vis, sample_text, asin

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		feature_idx = self.features[idx]
		userID = feature_idx[0][0]
		query = feature_idx[0][1]

		if self.is_training:
			pos_item = feature_idx[1][0]
			neg_items = feature_idx[1][1]

			pos_vis = self.vis_vec[pos_item]
			pos_text = self.text_vec[pos_item]

			neg_vis = [self.vis_vec[i] for i in neg_items]
			neg_text = [self.text_vec[i] for i in neg_items]
			neg_vis = np.array(neg_vis)
			neg_text = np.array(neg_text)

			sample = {'userID': userID, 'query': query,
					  'pos_vis': pos_vis, 'pos_text': pos_text,
					  'neg_vis': neg_vis, 'neg_text': neg_text}

		else:
			reviewerID = feature_idx[1][0]
			item = feature_idx[1][1]
			query_text = feature_idx[2]

			sample = {'userID': userID, 'query': query,
					  'reviewerID': reviewerID, 'item': item,
					  'query_text': query_text}
		return sample
