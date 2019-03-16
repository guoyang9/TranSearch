# TranSearch

A pytorch and tensorflow GPU implementation for our paper:

Yangyang Guo, Zhiyong Cheng, Liqiang Nie, Xin-Shun Xu, Mohan Kankanhalli (2018). Multi-modal Preference Modeling for Product Search. In Proceedings of ACM MM 2018.

**Please cite our MM'18 paper if you use our codes. Thanks!**

Please download the Amazon Dataset from http://jmcauley.ucsd.edu/data/amazon.

## The requirements are as follows:
1. python 3.5

2. pytorch 0.4.0

3. tensorflow 1.7

## Example to run:
1. Make sure the raw data, stopwords data are in the same direction.

2. Preprocessing data. Noted that the category >Clothing contains both the >Men's Clothing and >Women's Clothing. Therefore, before changed from other datasets to >Clothing dataset, remember to comment line 115 and uncomment line 116.
```
python script/extract.py --data_dir --count --is_clothing --img_feature_file
```

3. We leverage the PV-DM model to convert queries and product representations to the same latent space.
```
python script/doc2vec_img2dict.py --dataset --img_feature_file
```

4. Move the processed folder to the main model folder. 

Pre-training:
```
python pytorch/pre_train.py --embed_size --neg_number=20
```

5. Train and test the framework.
```
python pytorch/TranSearch.py --mode='text' --embed_size
```



