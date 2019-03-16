import struct

import numpy as np

def rm_items_without_image(df, img_path):
	asin_set = set(df.asin.unique())
	asin_candidate = set()
	with open(img_path, 'rb') as f:
		while True:
			item = f.read(10)
			if not item: break
			item = item.decode("utf-8")
			if item in asin_set:
				asin_candidate.add(item)
			f.read(4096*4)

	is_in_df = []
	for i in range(len(df)):
		if df['asin'][i] in asin_candidate:
			is_in_df.append('True')
		else:
			is_in_df.append('False')
	df['is_in_df'] = is_in_df
	df = df[df['is_in_df'] == 'True'].reset_index(drop=True)
	del(df['is_in_df'])

	return df


def get_img_feature(asin_set, img_path):
	image_feature_dict = {}
	with open(img_path, 'rb') as f:
		while True:
			asin = f.read(10)
			if not asin: break
			asin = asin.decode("utf-8")
			if asin in asin_set:
				feature = []
				for i in range(4096):
					feature.append(struct.unpack('f', f.read(4))[0])
				image_feature_dict[asin] = np.array(feature, dtype=np.float32)
			else:
				f.read(4096*4)

	return image_feature_dict
