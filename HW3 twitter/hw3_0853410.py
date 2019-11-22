import pandas as pd
import numpy as np


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
            print(i)
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

# sklearn has default English stop word

