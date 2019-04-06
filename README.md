# TranSearch

A pytorch and tensorflow GPU implementation for our paper:

Yangyang Guo, Zhiyong Cheng, Liqiang Nie, Xin-Shun Xu, Mohan Kankanhalli (2018). Multi-modal Preference Modeling for Product Search. In Proceedings of ACM MM 2018.

**Please cite our MM'18 paper if you use our codes. Thanks!**

You can download the Amazon Dataset from http://jmcauley.ucsd.edu/data/amazon.

## The requirements are as follows:
* python==3.6

* pandas==0.24.2

* numpy==1.16.2

* pytorch==0.4.0

* tensorflow==1.7

* gensim==3.7.1

* tensorboardX==1.6

## Example to run:
* Make sure the review data, meta data and image features are in the same directory.

* Preprocessing data. Noted that the category >Clothing contains both the >Men's Clothing and >Women's Clothing. 
	```
	python script/preprocess.py  --count=5 --is_clothing
	```

* We leverage the PV-DM model to convert queries and product representations to the same latent space. Besides, we extract the image features here. Be sure to use the correct image feature weights.
	```
	python script/doc2vec_img2dict.py  --img_feature_file
	```

* Pre-training is optional to train the final model. 
	```
	python pytorch/pre_train.py --embed_size=128 --neg_number=20
	```

* Train and test the framework.
	```
	python pytorch/TranSearch.py --mode='text' --embed_size=128
	```
