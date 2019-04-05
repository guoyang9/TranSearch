from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json

import numpy as np
import pandas as pd
import tensorflow as tf 
from gensim.models.doc2vec import Doc2Vec

import config


class PretrainData(object):
	def __init__(self, neg_number):
		""" For pretraining the image and review features."""
		self.neg_number = neg_number
		self.asin_dict = json.load(open(config.asin_sample_path, 'r'))
		self.all_items = set(self.asin_dict.keys())

		# textual feture data
		doc2vec_model = Doc2Vec.load(config.doc2model_path)
		self.text_vec = {
			asin: doc2vec_model.docvecs[asin] for asin in self.asin_dict}

		# visual feature data
		self.vis_vec = np.load(config.img_feature_path).item()

	def sample(self):
		""" Sample the anchor, positive, negative tuples."""
		self.features = []
		for asin in self.asin_dict:
			pos_items = self.asin_dict[asin]['positive']
			if not len(pos_items) == 0:
				for pos in pos_items:
					neg = np.random.choice(list(
								self.all_items - set(pos_items)),
								self.neg_number, replace=False)
					for n in neg:
						self.features.append((asin, pos, n))

	def build(self):
		self.sample()
		for pair in self.features:
			anchor_text = self.text_vec[pair[0]]
			anchor_vis = self.vis_vec[pair[0]]
			pos_text = self.text_vec[pair[1]]
			pos_vis = self.vis_vec[pair[1]]
			neg_text = self.text_vec[pair[2]]
			neg_vis = self.vis_vec[pair[2]]

			yield (anchor_text, anchor_vis,
					pos_text, pos_vis,
					neg_text, neg_vis) 


class TranSearchData(PretrainData):
	def __init__(self, neg_number, is_training):
		""" Without pre-train, input the raw data."""
		super().__init__(neg_number)
		self.is_training = is_training
		split = config.train_path if self.is_training else config.test_path
		self.data = pd.read_csv(split)
		self.query_dict = json.load(open(config.query_path, 'r'))
		self.user_bought = json.load(open(config.user_bought_path, 'r'))

	def sample_neg(self):
		""" Take the also_view or buy_after_viewing as negative samples. """
		self.features = []
		for i in range(len(self.data)):
			query_vec = self.doc2vec_model.docvecs[
								self.query_dict[self.data['query_'][i]]]
			if self.is_training:
				# We tend to sample negative ones from the also_view and 
				# buy_after_viewing items, if don't have enough, we then 
				# randomly sample negative ones.
				asin = self.data['asin'][i]
				sample = self.asin_dict[asin]
				all_sample = sample['positive'] + sample['negative']
				negs = np.random.choice(
					all_sample, self.neg_number, replace=False, p=sample['prob'])
			
				self.features.append(((
						self.data['userID'][i], query_vec), (asin, negs)))

			else:
				self.features.append(((self.data['userID'][i], query_vec),
					(self.data['reviewerID'][i], self.data['asin'][i]),
					self.data['query_'][i]))

	def get_all_test(self):
		for asin in self.asin_dict:
			sample_vis = np.reshape(self.vis_vec[asin], [1, -1])
			sample_text = np.reshape(self.text_vec[asin], [1, -1])

			yield sample_text, sample_vis, asin

	def get_instance(self):
		self.sample_neg()
		if self.is_training:
			for idx in range(len(self.features)):
				feature_idx = self.features[idx]
				userID = feature_idx[0][0]
				query = feature_idx[0][1]

				pos_item = feature_idx[1][0]
				neg_items = feature_idx[1][1]
				pos_vis = self.vis_vec[pos_item]
				pos_text = self.text_vec[pos_item]

				neg_vis = [self.vis_vec[n] for n in neg_items]
				neg_text = [self.text_vec[n] for n in neg_items]
				neg_vis = np.array(neg_vis)
				neg_text = np.array(neg_text)
				
				yield (userID, query,
						pos_text, pos_vis,
						neg_text, neg_vis)
		else:
			for idx in range(len(self.features)):
				feature_idx = self.features[idx]
				userID = np.array([feature_idx[0][0]])
				query = np.reshape(feature_idx[0][1], [1, -1])

				reviewerID = feature_idx[1][0]
				item = feature_idx[1][1]
				query_text = feature_idx[2]

				yield (userID, query, query_text, 
						item, reviewerID)
