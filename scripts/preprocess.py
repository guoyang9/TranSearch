import os, sys
import time
import gzip
import json
import argparse
import itertools
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np

import config
import text_process
import image_process


def get_df(path):
    """ Apply raw data to pandas DataFrame. """
    idx = 0
    df = {}
    g = gzip.open(path, 'rb')
    for line in g:
        df[idx] = eval(line)
        idx += 1
    return pd.DataFrame.from_dict(df, orient='index')


def extraction(meta_path, review_df, stop_words, count):
    """ Extracting useful infromation. """
    with gzip.open(meta_path, 'rb') as g:
        categories, also_viewed = {}, {}
        for line in g:
            line = eval(line)
            asin = line['asin']
            categories[asin] = line['categories']
            related = line['related'] if 'related' in line else None

            # fill the also_related dictionary
            also_viewed[asin] = []
            relations = ['also_viewed', 'buy_after_viewing']
            if related:
                also_viewed[asin] = [related[r] for r in relations if r in related]
                also_viewed[asin] = itertools.chain.from_iterable(also_viewed[asin])

    queries, reviews = [], []
    for i in range(len(review_df)):
        asin = review_df['asin'][i]
        review = review_df['reviewText'][i]
        category = categories[asin]

        # process queries
        qs = map(text_process._remove_dup, 
                    map(text_process._remove_char, category))
        qs = [[w for w in q if w not in stop_words] for q in qs]

        # process reviews
        review = text_process._remove_char(review)
        review = [w for w in review if w not in stop_words]

        queries.append(qs)
        reviews.append(review)

    review_df['query_'] = queries # write query result to dataframe

    # filtering words counts less than count
    reviews = text_process._filter_words(reviews, count)
    review_df['reviewText'] = reviews
    return review_df, also_viewed


def reindex(df):
    """ Reindex the reviewID from 0 to total length. """
    reviewer = df['reviewerID'].unique()
    reviewer_map = {r: i for i, r in enumerate(reviewer)}

    userIDs = [reviewer_map[df['reviewerID'][i]] for i in range(len(df))]
    df['userID'] = userIDs
    return df


def split_data(df):
    """ Enlarge the dataset with the corresponding user-query-item pairs."""
    df_enlarge = {}
    i = 0
    for row in range(len(df)):
        for q in df['query_'][row]:
            df_enlarge[i] = {'reviewerID': df['reviewerID'][row],
            'userID': df['userID'][row], 'query_': q,
            'asin': df['asin'][row], 'reviewText': df['reviewText'][row],
            'gender': df['gender'][row] if FLAGS.is_clothing else None}
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
            df_enlarge_test.reset_index(drop=True))


def get_user_bought(train_set):
    """ Obtain the products each user has bought before test. """
    user_bought = {}
    for i in range(len(train_set)):
        user = train_set['reviewerID'][i]
        item = train_set['asin'][i]
        if user not in user_bought:
            user_bought[user] = []
        user_bought[user].append(item)
    return user_bought


def rm_test(df, df_test):
    """ Remove test review data and remove duplicate reviews."""
    df = df.reset_index(drop=True)
    reviewText = []
    review_train_set = set()

    review_test = set(repr(
        df_test['reviewText'][i]) for i in range(len(df_test)))

    for i in range(len(df)):
        r = repr(df['reviewText'][i])
        if not r in review_train_set and not r in review_test:
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
    asin_samples = {}
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
        prob = np.array(pos_pr + neg_pr)
        prob = prob / prob.sum()

        asin_samples[asin] = {'positive': list(positive),
                                  'negative': negative,
                                  'prob': prob.tolist()}
    return asin_samples


def gender_split(df):
    """ Only useful when do experiments on clothing dataset. """
    gender = []
    for i in range(len(df)):
        q = list(itertools.chain.from_iterable(df['query_'][i]))
        if 'women' in q:
            gender.append('women')
        elif 'men' in q:
            gender.append('men')
        else:
            gender.append('other')
    df['gender'] = gender
    df_women = df[df['gender'] == 'women'].reset_index(drop=True)
    df_men = df[df['gender'] == 'men'].reset_index(drop=True)
    return df_women, df_men


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--review_file', 
        type=str,
        default='reviews_Toys_and_Games_5.json.gz',
        help="5-core review file")
    parser.add_argument('--meta_file', 
        type=str,
        default='meta_Toys_and_Games.json.gz',
        help="meta data file for the corresponding review file")
    parser.add_argument('--count', 
        type=int, 
        default=5,
        help="remove the words number less than count")
    parser.add_argument('--is_clothing', 
        action='store_true', 
        default=False,
        help="Clothing dataset needs to be split")
    parser.add_argument("--img_feature_file", 
        type=str,
        default="image_features_Toys_and_Games.b",
        help="the raw image feature file")
    global FLAGS
    FLAGS = parser.parse_args()


    ############################################# PREPARE PATHS #############################################
    if not os.path.exists(config.processed_path):
        os.makedirs(config.processed_path)

    stop_path = config.stop_file
    meta_path = os.path.join(config.main_path, FLAGS.meta_file)
    review_path = os.path.join(config.main_path, FLAGS.review_file)
    img_path = os.path.join(config.main_path, FLAGS.img_feature_file)

    review_df = get_df(review_path)
    review_df = image_process._rm_image(review_df, img_path) # remove items without image

    stop_df = pd.read_csv(stop_path, header=None, names=['stopword'])
    stop_words = set(stop_df['stopword'].unique())

    ############################################# PRE-EXTRACTION #############################################
    df, also_viewed = extraction(meta_path, review_df, stop_words, FLAGS.count)
    df = df.drop(['reviewerName', 'reviewTime', 'helpful', 'summary',
                'unixReviewTime', 'overall'], axis=1) # remove non-useful keys

    if FLAGS.is_clothing:
        df_women, df_men = gender_split(df)
        dataset = [('women_cloth', df_women), ('men_cloth', df_men)]
    else:
        dataset = [(config.dataset, df)]

    for d in dataset:
        df = reindex(d[1]) # reset the index of users
        print("The number of {} users is {:d}; items is {:d}; feedbacks is {:d}.".format(
            d[0], len(df.reviewerID.unique()), len(df.asin.unique()), len(df)))

        # sample negative items
        asin_samples = neg_sample(also_viewed, set(df.asin.unique()))
        print("Negtive samples of {} set done!".format(d[0]))
        json.dump(asin_samples, open(os.path.join(
                config.processed_path, '{}_asin_sample.json'.format(d[0])), 'w'))

        df, df_train, df_test = split_data(df)
        user_bought = get_user_bought(df_train)
        json.dump(user_bought, open(os.path.join(
                        config.processed_path, '{}_user_bought.json'.format(d[0])), 'w'))

        df = rm_test(df, df_test) # remove the reviews from test set
        df_train = rm_test(df_train, df_test)

        df.to_csv(os.path.join(
            config.processed_path, '{}_full.csv'.format(d[0])), index=False)
        df_train.to_csv(os.path.join(
            config.processed_path, '{}_train.csv'.format(d[0])), index=False)
        df_test.to_csv(os.path.join(
            config.processed_path, '{}_test.csv'.format(d[0])), index=False)


if __name__ == "__main__":
    main()
