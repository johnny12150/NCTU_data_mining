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
from sklearn.metrics import roc_curve, auc
import graphviz
import os

max_length = 150

arxiv = pd.read_csv('data/task2_trainset.csv')

feature = arxiv['Title']
feature2 = arxiv['Abstract']
feature3 = arxiv['Title']+arxiv['Abstract'].apply(lambda x: x.replace('$', ''))  # use both abstract and title as feature
target = arxiv['Task 2'].apply(lambda x: x.split()[0])
# target = arxiv['Task 2']  # multi-class target
target2 = arxiv['Categories'].apply(lambda x: x.split('/')[0])
print(target.value_counts().index)
print(target2.value_counts().index)
le = LabelEncoder()
target = le.fit_transform(target)
X_train, X_test, y_train, y_test = train_test_split(feature3, target, test_size=0.33, random_state=42)


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


def load_golve(token, save=False):
    # https://medium.com/@sarin.samarth07/glove-word-embeddings-with-keras-python-code-52131b0c8b1d
    # use golve pre-train embedding
    embedding_vector = {}
    f = open('glove.840B.300d.txt', encoding='utf8')
    for line in f:
        value = line.split(' ')
        word = value[0]
        coef = np.array(value[1:], dtype='float32')
        embedding_vector[word] = coef
    f.close()
    embedding_matrix = np.zeros((len(token.word_index)+1, 300))
    for word, i in token.word_index.items():
        embedding_value = embedding_vector.get(word)
        if embedding_value is not None:
            embedding_matrix[i] = embedding_value
    if save:
        np.save('golve_emb.npy', embedding_matrix)
    return embedding_matrix


X_train, X_test = preprocess(X_train, X_test, 8000)
# 組回句子
X_train = np.array([' '.join(x.tolist()) for x in X_train])
X_test = np.array([' '.join(x.tolist()) for x in X_test])
# build dictionary
tokenizer = Tokenizer(num_words=30000)
tokenizer.fit_on_texts(X_train)
# word to sequence
trainX_seq = tokenizer.texts_to_sequences(X_train)
testX_seq = tokenizer.texts_to_sequences(X_test)
# emb_matrix = load_golve(tokenizer)
emb_matrix = np.load('golve_emb.npy')
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


def CNN_1d(plot=False, classes=4, pretrain=True):
    model_CNN = Sequential()
    if pretrain:
        model_CNN.add(Embedding(output_dim=300, input_dim=len(tokenizer.word_index)+1, input_length=max_length, weights=[emb_matrix], trainable=False))
    else:
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


def transformer(classes=4, pretrain=True):
    S_inputs = Input(shape=(None,), dtype='int32')
    if pretrain:
        embeddings = Embedding(len(tokenizer.word_index), 128)(S_inputs)
        embeddings = Position_Embedding()(embeddings)  # 增加Position_Embedding能轻微提高准确率
    else:
        embeddings = Embedding(output_dim=300, input_dim=len(tokenizer.word_index) + 1, input_length=max_length,
                  weights=[emb_matrix], trainable=False)
    O_seq = Attention(32, 64)([embeddings, embeddings, embeddings])
    O_seq = GlobalAveragePooling1D()(O_seq)
    O_seq = Dropout(0.5)(O_seq)
    O_seq = Dense(units=200, activation='relu')(O_seq)
    outputs = Dense(classes, activation='softmax')(O_seq)
    model = Model(inputs=S_inputs, outputs=outputs)
    model.summary()
    return model


# model = rnn(classes=len(list(le.classes_)))
# model = lstm(classes=len(list(le.classes_)))
model = gru(classes=len(list(le.classes_)))
# model = CNN_1d(classes=len(list(le.classes_)))
# model = transformer(classes=len(list(le.classes_)))  # keras必須在2.1.6以前，之後API有改會報錯
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
train_history = model.fit(trainX_pad, y_train, batch_size=30, epochs=10, verbose=2, validation_split=0.2)
show_train_history(train_history, 'loss', 'val_loss')
show_train_history(train_history, 'acc', 'val_acc')
print(model.evaluate(testX_pad, y_test))


def roc(pred, label, classes=len(list(le.classes_))):
    pred_df = pd.get_dummies(pred).values
    y_le_df = pd.get_dummies(label).values
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(classes):
        fpr[i], tpr[i], threshold = roc_curve(y_le_df[:, i], pred_df[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    for i in range(classes):
        fig = plt.figure()
        lw = 2
        plt.plot(fpr[i], tpr[i], color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[i])
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic, Class="{}"'.format(list(le.classes_)[i]))
        plt.legend(loc="lower right")
        plt.show()


test_pred = model.predict_classes(testX_pad)
print(pd.crosstab(y_test, test_pred, rownames=['Label'], colnames=['Predict']))
roc(test_pred, y_test)
