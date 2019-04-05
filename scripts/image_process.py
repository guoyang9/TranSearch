import struct
import numpy as np


def _rm_image(df, img_path):
	""" remove items without images """
	asins = set(df.asin.unique())
	asins = _get_feature(asins, img_path).keys()

	keeps = ['True' if df['asin'][i] in asins else 'False'
			for i in range(len(df))]
	
	df['keeps'] = keeps
	df = df[df['keeps'] == 'True'].reset_index(drop=True)
	del(df['keeps'])
	return df


def _get_feature(asins, img_path):
	image_feature_dict = {}
	with open(img_path, 'rb') as f:
		while True:
			asin = f.read(10)
			if not asin: 
				break
			asin = asin.decode("utf-8")
			if asin in asins:
				feature = [struct.unpack('f', f.read(4))[0]
					for _ in range(4096)]
				image_feature_dict[asin] = np.array(
					feature, dtype=np.float32)
			else:
				f.read(4096*4)
	return image_feature_dict
