# choose dataset name
dataset = 'men_cloth'

# paths
main_path = '/home/share/guoyangyang/amazon/'
stop_file = './stopwords.txt'

processed_path = './processed/'

full_path = processed_path + '{}_full.csv'.format(dataset)
train_path = processed_path + '{}_train.csv'.format(dataset)
test_path = processed_path + '{}_test.csv'.format(dataset)

asin_sample_path = processed_path + '{}_asin_sample.json'.format(dataset)
user_bought_path = processed_path + '{}_user_bought.json'.format(dataset)

doc2model_path = processed_path + '{}_doc2model'.format(dataset)
query_path = processed_path + '{}_query.json'.format(dataset)
img_feature_path = processed_path + '{}_img_feature.npy'.format(dataset)

weights_path = processed_path + 'Variable/'
image_weights_path = weights_path + 'visual_FC.pt'
text_weights_path = weights_path + 'textual_FC.pt'

# embedding size
visual_size = 4096
textual_size = 512
