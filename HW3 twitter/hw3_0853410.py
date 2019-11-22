import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


def load_data(filename, type):
    # should use utf-8 encode to open
    f = open(filename, 'r', encoding="utf-8")
    data = f.readlines()
    data = np.asarray(data)
    # save comment and label
    twitter_arr = []
    twitter_label = []
    if type == 'train':
        symbol = ' +++$+++ '
    elif type == 'test':
        symbol = '#####'

    for line in data:
        # skip blank line (for test data)
        if line.rstrip():
            # split the symbol between feature and label
            label = line.split(symbol)[0]
            comment = line.split(symbol)[1].split('\n')[0]
            twitter_arr.append(comment)
            twitter_label.append(label)

    # transform array to df
    df = pd.DataFrame(twitter_label, columns=['label'])
    df2 = pd.DataFrame(twitter_arr, columns=['comment'])
    twitter_df = pd.concat([df, df2], axis=1)
    return twitter_df


train = load_data('./training_label.txt', 'train')
test = load_data('./testing_label.txt', 'test')


# NLP preprocess
def preprocess(data):
    # sklearn has default English stop word
    vectorizer = CountVectorizer()
    # this may take a while
    tmp = vectorizer.fit_transform(data)
    # return comment without stop words
    tmp = vectorizer.inverse_transform(tmp)
    # combine array to string/ sentence
    noStopWord = []
    for j in tmp:
        noStopWord.append(' '.join(j))

    tfidf = TfidfVectorizer()
    word_tf = tfidf.fit_transform(noStopWord)
    return word_tf


trainX = preprocess(train['comment'].values)


from xgboost.sklearn import XGBClassifier
from sklearn import metrics

# XGboost gpu (using sklearn interface), you need to be careful about the size of your VRAM
# clf = XGBClassifier(tree_method='gpu_hist', gpu_id=0)
# cpu
clf = XGBClassifier()

clf.fit(trainX, train['label'], eval_metric='auc')
print(clf.score(trainX, train['label']))
# y_pred = clf.predict(trainX)
# print(metrics.accuracy_score(train['label'], y_pred))

