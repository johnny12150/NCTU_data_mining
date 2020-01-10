import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import SimpleRNN
from keras.layers import CuDNNLSTM, GRU, Conv1D, GlobalMaxPooling1D, MaxPooling1D, GlobalAveragePooling1D, Input
from keras.layers.embeddings import Embedding
from keras.utils.vis_utils import plot_model
from attention import Position_Embedding, Attention
import graphviz
import os

max_length = 30

arxiv = pd.read_csv('data/task2_trainset.csv')

feature = arxiv['Title']
feature2 = arxiv['Abstract']
target = arxiv['Task 2'].apply(lambda x: x.split()[0])
target2 = arxiv['Categories'].apply(lambda x: x.split('/')[0])
print(target.value_counts().index)
print(target2.value_counts().index)
le = LabelEncoder()
target = le.fit_transform(target)
X_train, X_test, y_train, y_test = train_test_split(feature, target, test_size=0.33, random_state=42)


def preprocess(train, test, max_feature=3000, stop_word=True):
    if stop_word:
        vectorizer = TfidfVectorizer(stop_words='english', max_features=max_feature)
    else:
        vectorizer = TfidfVectorizer(max_features=max_feature)
    # this may take a while
    tmp = vectorizer.fit_transform(train)
    tmp2 = vectorizer.transform(test)
    # inverse back to normal words
    tmp = vectorizer.inverse_transform(tmp)
    tmp2 = vectorizer.inverse_transform(tmp2)
    return tmp, tmp2


X_train, X_test = preprocess(X_train, X_test, 8000)
# 組回句子
X_train = np.array([' '.join(x.tolist()) for x in X_train])
X_test = np.array([' '.join(x.tolist()) for x in X_test])
# build dictionary
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(X_train)
# word to sequence
trainX_seq = tokenizer.texts_to_sequences(X_train)
testX_seq = tokenizer.texts_to_sequences(X_test)
# padding
trainX_pad = sequence.pad_sequences(trainX_seq, maxlen=max_length)
testX_pad = sequence.pad_sequences(testX_seq, maxlen=max_length)


def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'])
    plt.show()


def rnn(plot=False, classes=4):
    model_RNN = Sequential()
    model_RNN.add(Embedding(output_dim=128, input_dim=len(tokenizer.word_index), input_length=max_length))
    model_RNN.add(SimpleRNN(60, return_sequences=True))
    model_RNN.add(Dropout(0.7))
    model_RNN.add(SimpleRNN(30))
    model_RNN.add(Dense(units=32, activation='relu'))
    model_RNN.add(Dense(units=classes, activation='softmax'))
    print(model_RNN.summary())
    if plot:
        plot_model(model_RNN, to_file='model_RNN.png', show_shapes=True)
    return model_RNN


def lstm(plot=False, classes=4):
    model_LSTM = Sequential()
    model_LSTM.add(Embedding(output_dim=128, input_dim=len(tokenizer.word_index), input_length=max_length))
    model_LSTM.add(Dropout(0.4))
    model_LSTM.add(CuDNNLSTM(32, return_sequences=True))
    model_LSTM.add(CuDNNLSTM(32, return_sequences=True))
    model_LSTM.add(CuDNNLSTM(32, return_sequences=False))
    model_LSTM.add(Dense(units=50, activation='relu'))
    model_LSTM.add(Dropout(0.4))
    model_LSTM.add(Dense(units=classes, activation='softmax'))
    model_LSTM.summary()
    if plot:
        plot_model(model_LSTM, to_file='model_LSTM.png', show_shapes=True)
    return model_LSTM


def gru(plot=False, classes=4):
    model_GRU = Sequential()
    model_GRU.add(Embedding(output_dim=128, input_dim=len(tokenizer.word_index), input_length=max_length))
    model_GRU.add(Dropout(0.4))
    model_GRU.add(GRU(32, return_sequences=True))
    model_GRU.add(GRU(32, return_sequences=False))
    model_GRU.add(Dense(units=50, activation='relu'))
    model_GRU.add(Dropout(0.4))
    model_GRU.add(Dense(units=classes, activation='softmax'))
    model_GRU.summary()
    if plot:
        plot_model(model_GRU, to_file='model_GRU.png', show_shapes=True)
    return model_GRU


def CNN_1d(plot=False, classes=4):
    model_CNN = Sequential()
    model_CNN.add(Embedding(output_dim=64, input_dim=len(tokenizer.word_index), input_length=max_length))
    model_CNN.add(Conv1D(64, 5, activation='relu'))
    model_CNN.add(MaxPooling1D())
    model_CNN.add(Conv1D(64, 5, activation='relu'))
    model_CNN.add(GlobalMaxPooling1D())
    model_CNN.add(Dense(classes, activation='softmax'))
    model_CNN.summary()
    if plot:
        plot_model(model_CNN, to_file='model_CNN.png', show_shapes=True)
    return model_CNN


def transformer(classes=4):
    S_inputs = Input(shape=(None,), dtype='int32')
    embeddings = Embedding(len(tokenizer.word_index), 128)(S_inputs)
    embeddings = Position_Embedding()(embeddings)  # 增加Position_Embedding能轻微提高准确率
    O_seq = Attention(8, 16)([embeddings, embeddings, embeddings])
    O_seq = GlobalAveragePooling1D()(O_seq)
    O_seq = Dropout(0.5)(O_seq)
    O_seq = Dense(units=50, activation='relu')(O_seq)
    outputs = Dense(classes, activation='softmax')(O_seq)
    model = Model(inputs=S_inputs, outputs=outputs)
    model.summary()
    return model


# model = rnn()
# model = lstm()
# model = gru()
# model = CNN_1d()
model = transformer()  # keras必須在2.1.6以前，之後API有改會報錯
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
train_history = model.fit(trainX_pad, y_train, batch_size=30, epochs=30, verbose=2, validation_split=0.2)
show_train_history(train_history, 'loss', 'val_loss')
show_train_history(train_history, 'acc', 'val_acc')
print(model.evaluate(testX_pad, y_test))



