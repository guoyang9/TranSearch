import os, argparse
import random, json
import collections

import numpy as np
import pandas as pd
from ast import literal_eval
from gensim.models import doc2vec

import image_feature_process as ifp

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", default='MenClothing', 
			type=str, help="choose dataset to process.")
parser.add_argument("--embedding_size", default=512, type=int,
			help="doc dimension.")
parser.add_argument("--window_size", default=3, type=int,
			help="sentence window size.")
parser.add_argument("--img_feature_file", 
			default="img_features_Clothing.b",
 			type=str, help="the raw image feature file.")

FLAGS = parser.parse_args()

ROOT_DIR = '../processed/'
IMG_ROOT_DIR = '/media/yang/DATA/Datasets/amazon'

full_file = os.path.join(ROOT_DIR, FLAGS.dataset, 'full.csv')
full_data = pd.read_csv(full_file)
full_data.query_ = full_data.query_.apply(literal_eval)
full_data.reviewText = full_data.reviewText.apply(literal_eval)
asin_set = set(full_data.asin.unique())
img_path = os.path.join(IMG_ROOT_DIR, FLAGS.img_feature_file)

#Gather reviews to same asins.
rawDoc = collections.defaultdict(list)
for k, v in zip(full_data.asin, full_data.reviewText):
	rawDoc[k].append(v)

#Concatenate the reviews together.
for k in rawDoc.keys():
	m = []
	for i in rawDoc[k]:
		m.extend(i)
	rawDoc[k] = m

#For query, it's hard to tag, so we just random tag them.
query_idx = 0
query_dic = {} #For query index and doc2vec index matching.
for q in full_data['query_']:
	if repr(q) not in query_dic:
			query_dic[repr(q)] = query_idx
			rawDoc[query_idx] = q
			query_idx += 1

##################################### Start Model #############
docs = []
analyzedDocument = collections.namedtuple(
							'AnalyzedDocument', 'words tags')
for d in rawDoc.keys():
	docs.append(analyzedDocument(rawDoc[d], [d]))

alpha_val = 0.025
min_alpha_val = 1e-4
passes = 40

alpha_delta = (alpha_val - min_alpha_val) / (passes - 1)

# model = doc2vec.Doc2Vec(vector_size=FLAGS.embedding_size, dm=1, 
# 						sample=1e-3, negative=5, 
# 						window=2, min_count=0, 
# 						workers=4, epochs=10)

model = doc2vec.Doc2Vec(vector_size=FLAGS.embedding_size,
						window=FLAGS.window_size, min_count=2, 
						workers=4, epochs=10)

model.build_vocab(docs) # Building vocabulary

for epoch in range(passes):
    random.shuffle(docs)

    model.alpha, model.min_alpha = alpha_val, alpha_val
    model.train(docs, total_examples=len(docs), epochs=model.iter)

    alpha_val -= alpha_delta

#Save model
model.save(os.path.join(ROOT_DIR, FLAGS.dataset, 'doc2vecFile'))

#Write query embedding and index to disk
json.dump(query_dic, open(os.path.join(
				ROOT_DIR, FLAGS.dataset, 'queryFile.json'), 'w'))
print("The query number is %d." %len(query_dic))

#Write image feature to disk
image_feature_dict = ifp.get_img_feature(asin_set, img_path)
np.save(os.path.join(ROOT_DIR, FLAGS.dataset, 
						'image_feature.npy'), image_feature_dict)
