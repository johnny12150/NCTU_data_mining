import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from keras.utils.vis_utils import plot_model
# import graphviz
# import os
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import SimpleRNN
from keras.layers import CuDNNLSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence


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
# os.environ["PATH"] += os.pathsep + 'C:/Users/Wade/Anaconda3/Library/bin/graphviz'
max_length = 10


# NLP preprocess
def preprocess(train, test, type='stop word'):
    if type == 'stop word':
        vectorizer = CountVectorizer(stop_words='english')
        # this may take a while
        tmp = vectorizer.fit_transform(train)
        tmp2 = vectorizer.transform(test)
        # inverse back to normal words
        tmp = vectorizer.inverse_transform(tmp)
        tmp2 = vectorizer.inverse_transform(tmp2)
        return tmp, tmp2
    else:
        # max_features 決定最多幾個字
        tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
        word_tf = tfidf.fit_transform(train)
        word_tf2 = tfidf.transform(test)
        return word_tf, word_tf2, tfidf.vocabulary_


def metric_acc(model, x, y):
    y_pred = model.predict(x)
    report = classification_report(y, y_pred, output_dict=True)
    return pd.DataFrame.from_dict(report).T


def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='center right')
    plt.show()


trainX, testX = preprocess(train['comment'], test['comment'])
# 組回句子
trainX = np.array([' '.join(x.tolist()) for x in trainX])
testX = np.array([' '.join(x.tolist()) for x in testX])
# build dictionary
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(trainX)
print(len(tokenizer.word_index))
trainX_seq = tokenizer.texts_to_sequences(trainX)
testX_seq = tokenizer.texts_to_sequences(testX)
# 計算去除stop word後句子的平均長度
print(sum([len(x) for x in trainX_seq])/ len(trainX_seq))
# padding
trainX_pad = sequence.pad_sequences(trainX_seq, maxlen=max_length)
testX_pad = sequence.pad_sequences(testX_seq, maxlen=max_length)

# 也可以選擇做TFIDF
trainX_tfidf, testX_tfidf, words = preprocess(train['comment'].values, test['comment'].values, 'tfidf')
# 從spare matrix轉回np array
trainX_tfidf = trainX_tfidf.toarray()
testX_tfidf = testX_tfidf.toarray()
# 轉成3D讓RNN/ Embedding能吃
# https://stackoverflow.com/questions/52182185/how-to-use-tf-idf-vectorizer-with-lstm-in-keras-python
trainX_tfidf = trainX_tfidf[:, :, None]
testX_tfidf = testX_tfidf[:, :, None]


# 1. RNN, 沒有gpu版所以會超慢
def rnn(x, y, emb=1):
    model_RNN = Sequential()
    if emb:
        model_RNN.add(Embedding(output_dim=128, input_dim=len(tokenizer.word_index), input_length=max_length))
    else:
        model_RNN.add(SimpleRNN(64, input_shape=trainX.shape[1:], return_sequences=True))  # return_sequences = True 才能疊多層RNN
    model_RNN.add(SimpleRNN(60, return_sequences=True))
    model_RNN.add(Dropout(0.7))
    model_RNN.add(SimpleRNN(30))
    model_RNN.add(Dense(units=32, activation='relu'))
    model_RNN.add(Dense(units=1, activation='sigmoid'))
    print(model_RNN.summary())
    # plot_model(model_RNN, to_file='model_RNN.png', show_shapes=True)
    model_RNN.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    train_history_RNN = model_RNN.fit(x, y, batch_size=100, epochs=10, verbose=2, validation_split=0.2)
    show_train_history(train_history_RNN, 'loss', 'val_loss')
    show_train_history(train_history_RNN, 'acc', 'val_acc')
    return model_RNN, train_history_RNN


model_r, history_r = rnn(trainX_pad, train['label'])
print(model_r.evaluate(testX_pad, test['label']))
# baseline acc
print(np.sum(test['label'].values.astype(int))/ 90)

# TFIDF維度過大容易導致ram爆炸且一個epoch要三小時
# model_r2 = rnn(trainX_tfidf, train['label'], 0)


# 2. LSTM, 使用gpu版
def lstm(x, y, emb=1):
    model_LSTM = Sequential()
    if emb:
        model_LSTM.add(Embedding(output_dim=128, input_dim=len(tokenizer.word_index), input_length=max_length))
        model_LSTM.add(CuDNNLSTM(32, return_sequences=True))
    else:
        model_LSTM.add(CuDNNLSTM(32, input_shape=trainX.shape[1:], return_sequences=True))
    model_LSTM.add(Dropout(0.4))
    # model_LSTM.add(CuDNNLSTM(32, return_sequences=True))
    model_LSTM.add(CuDNNLSTM(32, return_sequences=False))
    model_LSTM.add(Dense(units=50, activation='relu'))
    model_LSTM.add(Dropout(0.4))
    model_LSTM.add(Dense(units=1, activation='sigmoid'))
    print(model_LSTM.summary())
    # plot_model(model_LSTM, to_file='model_LSTM.png', show_shapes=True)
    model_LSTM.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    train_history_LSTM = model_LSTM.fit(x, y, batch_size=200, epochs=10, verbose=2, validation_split=0.2)
    show_train_history(train_history_LSTM, 'loss', 'val_loss')
    show_train_history(train_history_LSTM, 'accuracy', 'val_accuracy')

    return model_LSTM, train_history_LSTM


model_ls, history_ls = lstm(trainX_pad, train['label'])
print(model_ls.evaluate(testX_pad, test['label']))
# 多層LSTM會出現錯誤, https://github.com/keras-team/keras/issues/12206
