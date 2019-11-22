import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost.sklearn import XGBClassifier
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier


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

clf = AdaBoostClassifier(n_estimators=100, random_state=0)  # training acc = 0.719
clf.fit(trainX, train['label'])
print(clf.score(trainX, train['label']))

# XGboost gpu (using sklearn interface)
# you need to be careful about the size of your VRAM (Process finished with exit code -1073740791 (0xC0000409))
# clf = XGBClassifier(tree_method='gpu_hist', gpu_id=0, n_estimators=100, learning_rate=0.3, max_depth=6)
# clf.fit(trainX[:1000], train['label'].values[:1000], eval_metric='auc')

# batch train for gpu
params = {'tree_method': 'gpu_hist', 'gpu_id': 0}
# https://stackoverflow.com/questions/38079853/how-can-i-implement-incremental-training-for-xgboost

# cpu
clf = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6)  # acc on train = 0.73
# clf.fit(trainX, train['label'], eval_metric='auc')
# print(clf.score(trainX, train['label']))

# alternative way of evaluate training accuracy
# y_pred = clf.predict(trainX)
# print(metrics.accuracy_score(train['label'], y_pred))

