import pandas as pd
import numpy as np

import gzip, json
import argparse, os, time, sys
import csv

import text_preprocess as tp
import image_feature_process as ifp


def getDF(path):
    """Apply raw data to pandas DataFrame."""
    i = 0
    df = {}
    g = gzip.open(path, 'rb')
    for line in g:
        df[i] = eval(line)
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')

def index_set(s):
    """Apply complex index to numerical format."""
    i = 0
    s_map = {}
    for key in s:
        s_map[key] = i
        i += 1
    return s_map

def extraction(meta_path, reviewDF, stop_path, count):
    """Extract item categories from meta data."""
    with gzip.open(meta_path, 'rb') as g:
        category_dic = {}
        also_viewed = {}
        for line in g:
            line = eval(line)
            category_dic[line['asin']] = line['categories']

            if (not 'related' in line) or (not 'also_viewed' in line['related'] and
                not 'buy_after_viewing' in line['related']):
                also_viewed[line['asin']] = []
            elif 'also_viewed' in line['related'] and not 'buy_after_viewing' in line['related']:
                also_viewed[line['asin']] = line['related']['also_viewed']
            elif not 'also_viewed' in line['related'] and 'buy_after_viewing' in line['related']:
                also_viewed[line['asin']] = line['related']['buy_after_viewing']
            else:
                also_viewed[line['asin']] = line['related']['also_viewed']
                also_viewed[line['asin']].extend(line['related']['buy_after_viewing'])

    #For further stop words processing
    stopDF = pd.read_csv(stop_path, header=None, names=['stopword'])
    stop_set = set(stopDF['stopword'].unique())

    query_list = []
    review_text_list = []
    review_with_des = []
    for i in range(len(reviewDF)):
        asin = reviewDF.asin[i]
        value_queries = category_dic[asin]

        #Remove the categories with ony one level.
        #Remove punctuation marks, duplicated and stop words.
        new_queries = []
        for value_query in value_queries:
            if len(value_query) == 1:
                continue
            value_query = tp.remove_char(value_query)
            value_query = tp.remove_dup(value_query)
            value_query = tp.remove_stop(value_query, stop_set)
            new_queries.append(value_query)
        if len(new_queries) == 0:
            value_query = value_queries[0]
            value_query = tp.remove_char(value_query)
            value_query = tp.remove_dup(value_query)
            value_query = tp.remove_stop(value_query, stop_set)
            new_queries.append(value_query)

        #Process reviewText and description.
        review_text = reviewDF.reviewText[i]
        review_text = tp.remove_char(review_text)
        review_text = tp.remove_stop(review_text, stop_set)

        query_list.append(new_queries)
        review_text_list.append(review_text)

    reviewDF['query_'] = query_list #Write query result to dataframe

    #Filtering words counts less than count
    review_text_list = tp.filter_words(review_text_list, count)
    reviewDF['reviewText'] = review_text_list

    return reviewDF, set(reviewDF.asin.unique()), also_viewed

def u_i_reindex(df):
    """Reindex the reviewID."""
    #create id_to_number map
    reviewerID_set = set(df['reviewerID'].unique())
    reviewerID_map = index_set(reviewerID_set)

    userID_list = []
    for i in range(len(df)):
        reviewerID_key = df['reviewerID'][i]
        userID_list.append(reviewerID_map[reviewerID_key])
    df['userID'] = userID_list

    return df

def split_data(df):
    """Enlarge the dataset with the corresponding user-query-item pairs."""
    df_enlarge = {}
    i = 0
    for row in range(len(df)):
        for q in df['query_'][row]:
            df_enlarge[i] = {'reviewerID': df['reviewerID'][row],
            'userID': df['userID'][row], 'query_': q,
            # 'asin': df['asin'][row], 'gender': df['gender'][row],
            'asin': df['asin'][row],
            'reviewText': df['reviewText'][row]}
            i += 1

    df_enlarge = pd.DataFrame.from_dict(df_enlarge, orient='index')

    split_filter = []
    df_enlarge = df_enlarge.sort_values(by='userID')
    user_length = df_enlarge.groupby('userID').size().tolist()

    for user in range(len(user_length)):
        length = user_length[user]
        tag = ['Train' for _ in range(int(length * 0.8))]
        tag_test = ['Test' for _ in range(length - int(length * 0.8))]
        tag.extend(tag_test)
        if length == 1:
            tag = ['Train']
        tag = np.random.choice(tag, length, replace=False)
        split_filter.extend(tag.tolist())

    df_enlarge['filter'] = split_filter
    df_enlarge_train = df_enlarge[df_enlarge['filter'] == 'Train']
    df_enlarge_test = df_enlarge[df_enlarge['filter'] == 'Test']

    return (df_enlarge.reset_index(drop=True),
            df_enlarge_train.reset_index(drop=True),
            df_enlarge_test.reset_index(drop=True)
            )

def get_user_bought(train_set):
    """ obtain the products each user has bought before test. """
    user_bought = {}
    for i in range(len(train_set)):
        user = train_set['reviewerID'][i]
        item = train_set['asin'][i]
        if user not in user_bought:
            user_bought[user] = []
        user_bought[user].append(item)

    return user_bought

def removeTest(df, df_test):
    """Remove test review data and remove duplicate."""
    df = df.reset_index(drop=True)
    reviewText = []
    review_train_set = set()
    review_test_set = set()

    for i in range(len(df_test)):
        review_test_set.add(repr(df_test['reviewText'][i]))

    for i in range(len(df)):
        r = repr(df['reviewText'][i])
        if not r in review_train_set and not r in review_test_set:
            review_train_set.add(r)
            reviewText.append(df['reviewText'][i])
        else:
            reviewText.append("[]")
    df['reviewText'] = reviewText

    return df

def neg_sample(also_viewed, unique_asin):
    """
    Sample the negative set for each asin(item), first add the 'also_view'
    asins to the dict, then add asins share the same query.
    """
    asin_sample_dict = {}
    for asin in unique_asin:
        positive = set([a for a in also_viewed[asin] if a in unique_asin])
        negative = list(unique_asin - positive)
        if not len(positive) < 20:
            negative = np.random.choice(
                        negative, 5*len(positive), replace=False).tolist()

        elif not len(positive) < 5:
            negative = np.random.choice(
                        negative, 10*len(positive), replace=False).tolist()

        elif not len(positive) < 1:
            negative = np.random.choice(
                         negative, 20*len(positive), replace=False).tolist()

        else:
            negative = np.random.choice(negative, 50, replace=False).tolist()

        pos_pr = [0.7 for _ in range(len(positive))]
        neg_pr = [0.3 for _ in range(len(negative))]
        pos_pr.extend(neg_pr)
        pos_pr = np.array(pos_pr)
        pos_pr = pos_pr / pos_pr.sum()


        asin_sample_dict[asin] = {'positive': list(positive),
                                  'negative': negative,
                                  'prob': pos_pr.tolist()}

    return asin_sample_dict

def w_m_split(df):
    """Only useful when do experiments on clothing dataset."""
    gender = []
    for i in range(len(df)):
        for (idx, q) in enumerate(df['query_'][i]):
            if 'women' in q:
                gender.append('women')
                break
            elif 'men' in q:
                gender.append('men')
                break
            if idx == len(df['query_'][i]) - 1:
                gender.append('other')

    df['gender'] = gender
    df_women = df[df['gender'] == 'women'].reset_index(drop=True)
    df_men = df[df['gender'] == 'men'].reset_index(drop=True)

    return df_women, set(df_women.asin.unique()), df_men, set(df_men.asin.unique())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
        default='/media/yang/DATA/Datasets/amazon',
        help="All source files should be under this folder.")
    parser.add_argument('--review_file', type=str,
        default='reviews_Clothing_Shoes_and_Jewelry_5.json.gz',
        help="5 core review file.")
    parser.add_argument('--meta_file', type=str,
        default='meta_Clothing_Shoes_and_Jewelry.json.gz',
        help="Meta data file for the corresponding review file.")
    parser.add_argument('--count', type=int, default=5,
        help="Remove the words number less than count.")
    parser.add_argument('--stop_file', type=str, default='stopwords.txt',
        help="Stop words file.")
    parser.add_argument('--save_path', type=str, default='./processed',
        help="Destination to save all the files.")
    parser.add_argument('--is_clothing', action='store_true', default=False,
        help="Clothing dataset needs to be split.")
    parser.add_argument("--img_feature_file", default="img_features_Clothing.b",
                    type=str, help="the raw image feature file.")

    FLAGS = parser.parse_args()

    if not os.path.exists(FLAGS.save_path):
        os.makedirs(FLAGS.save_path)

    meta_path = os.path.join(FLAGS.data_dir, FLAGS.meta_file)
    review_path = os.path.join(FLAGS.data_dir, FLAGS.review_file)
    stop_path = os.path.join(FLAGS.data_dir, FLAGS.stop_file)

    reviewDF = getDF(review_path)

    #Remove items which don't have image.
    img_path = os.path.join(FLAGS.data_dir, FLAGS.img_feature_file)
    reviewDF = ifp.rm_items_without_image(reviewDF, img_path)

    df, unique_asin, also_view_dict = extraction(
                                meta_path, reviewDF, stop_path, FLAGS.count)
    print("Pre extraction done!\nThe number of users is %d.\tNumber of items is %d.\tNumber of feedbacks is %d." %(
        len(df.reviewerID.unique()), len(unique_asin), len(df)))
    df = df.drop(['reviewerName', 'reviewTime', 'helpful', 'summary',
                                'unixReviewTime', 'overall'], axis=1)

    if not FLAGS.is_clothing:
        df = reviewID_reindex(df) #Reset the index of users.

        asin_sample_dict = neg_sample(also_view_dict, unique_asin)
        print("Negtive samples set up done!")
        json.dump(asin_sample_dict, open(os.path.join(
                                    FLAGS.save_path, 'asin_sample.json'), 'w'))

        df, df_train, df_test, = split_data(df)
        print("Dataset splitting done!")
        user_bought = get_user_bought(df_train)
        json.dump(user_bought, open(os.path.join(
                                    FLAGS.save_path, 'user_bought.json'), 'w'))
        df = removeTest(df, df_test) #Remove the reviews from test set.
        df_train = removeTest(df_train, df_test)

        df.to_csv(os.path.join(FLAGS.save_path, 'full.csv'), index=False)
        df_train.to_csv(os.path.join(FLAGS.save_path, 'train.csv'), index=False)
        df_test.to_csv(os.path.join(FLAGS.save_path, 'test.csv'), index=False)

    else:
        df_women, women_asin, df_men, men_asin = w_m_split(df)
        df_women = reviewID_reindex(df_women)
        df_men = reviewID_reindex(df_men)
        print("Pre extraction done!\nThe number of women users is %d.\tNumber of items is %d.\tNumber of feedbacks is %d." %(
            len(df_women.reviewerID.unique()), len(women_asin), len(df_women)))
        print("Pre extraction done!\nThe number of men users is %d.\tNumber of items is %d.\tNumber of feedbacks is %d." %(
            len(df_men.reviewerID.unique()), len(men_asin), len(df_men)))

        asin_sample_women = neg_sample(also_view_dict, women_asin)
        asin_sample_men = neg_sample(also_view_dict, men_asin)
        print("Negtive samples set up done!")
        json.dump(asin_sample_women, open(os.path.join(
                            FLAGS.save_path, 'asin_sample_women.json'), 'w'))
        json.dump(asin_sample_men, open(os.path.join(
                            FLAGS.save_path, 'asin_sample_men.json'), 'w'))

        df_women, df_women_train, df_women_test = split_data(df_women)
        user_bought = get_user_bought(df_women_train)
        json.dump(user_bought, open(os.path.join(
                            FLAGS.save_path, 'user_bought_women.json'), 'w'))
        df_women = removeTest(df_women, df_women_test)
        df_women_train = removeTest(df_women_train, df_women_test)

        df_men, df_men_train, df_men_test = split_data(df_men)
        user_bought = get_user_bought(df_men_train)
        json.dump(user_bought, open(os.path.join(
                            FLAGS.save_path, 'user_bought_men.json'), 'w'))
        df_men = removeTest(df_men, df_men_test)
        df_men_train = removeTest(df_men_train, df_men_test)
        print("Dataset splitting done!")

        df_women.to_csv(os.path.join(FLAGS.save_path, 'women_full.csv'), index=False)
        df_women_train.to_csv(os.path.join(FLAGS.save_path, 'women_train.csv'), index=False)
        df_women_test.to_csv(os.path.join(FLAGS.save_path, 'women_test.csv'), index=False)

        df_men.to_csv(os.path.join(FLAGS.save_path, 'men_full.csv'), index=False)
        df_men_train.to_csv(os.path.join(FLAGS.save_path, 'men_train.csv'), index=False)
        df_men_test.to_csv(os.path.join(FLAGS.save_path, 'men_test.csv'), index=False)

    print("All processes done!")


if __name__ == "__main__":
    main()
