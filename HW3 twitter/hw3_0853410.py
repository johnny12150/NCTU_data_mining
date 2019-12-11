import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report
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
def preprocess(train, test):
    # sklearn has default English stop word
    vectorizer = CountVectorizer(stop_words='english')
    # this may take a while
    tmp = vectorizer.fit_transform(train)
    tmp2 = vectorizer.transform(test)
    # return comment without stop words, alternative is using the tfidf_transformer
    tmp = vectorizer.inverse_transform(tmp)
    tmp2 = vectorizer.inverse_transform(tmp2)
    # combine array to string/ sentence
    noStopWord = []
    noStopWord2 = []
    for j in tmp:
        noStopWord.append(' '.join(j))

    for j in tmp2:
        noStopWord2.append(' '.join(j))

    tfidf = TfidfVectorizer(stop_words='english')
    word_tf = tfidf.fit_transform(train)
    word_tf2 = tfidf.transform(test)
    return word_tf, word_tf2


def metric_acc(model, x, y):
    y_pred = model.predict(x)
    report = classification_report(y, y_pred, output_dict=True)
    return pd.DataFrame.from_dict(report).T


trainX, testX = preprocess(train['comment'].values, test['comment'].values)

est = [50, 100, 150, 200, 300, 400, 500]
for k in est:
    clf = AdaBoostClassifier(n_estimators=k, random_state=0)  # training acc = 0.719
    clf.fit(trainX, train['label'])
    ad_training_result = metric_acc(clf, trainX, train['label'])
    ad_testing_result = metric_acc(clf, testX, test['label'])
    print(clf.score(trainX, train['label']))
    print(clf.score(testX, test['label']))
    print('--'*6)

# XGboost gpu (using sklearn interface)
# you need to be careful about the size of your VRAM (Process finished with exit code -1073740791 (0xC0000409))
# clf = XGBClassifier(tree_method='gpu_hist', gpu_id=0, n_estimators=100, learning_rate=0.3, max_depth=6)
# clf.fit(trainX[:1000], train['label'].values[:1000], eval_metric='auc')

# batch train for gpu
params = {'tree_method': 'gpu_hist', 'gpu_id': 0}
# https://stackoverflow.com/questions/38079853/how-can-i-implement-incremental-training-for-xgboost

est = [50, 100, 150, 200, 300, 400, 500]
for k in est:
    clf = XGBClassifier(n_estimators=k, learning_rate=0.1, max_depth=6)  # acc on train = 0.73
    clf.fit(trainX, train['label'], eval_metric='auc')
    print(clf.score(trainX, train['label']))
    print(clf.score(testX, test['label']))
    training_result = metric_acc(clf, trainX, train['label'])
    testing_result = metric_acc(clf, testX, test['label'])
    print('--'*6)

