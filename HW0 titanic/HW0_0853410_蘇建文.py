import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier

# dataset/ csv is from kaggle: https://www.kaggle.com/c/titanic
DF = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
all = DF.append(test)

# 特徵選取
# train_feature = DF[['Sex', 'Embarked', 'Pclass', 'SibSp', 'Parch']].copy()
train_feature = DF[['Sex', 'Pclass']].copy()
train_labels = DF['Survived'].copy()

# 處理空值, 用眾數補embarked
# train_feature['Embarked'] = train_feature['Embarked'].fillna(train_feature['Embarked'].mode()[0])

# 嘗試新feature來增加 model準確度
# 1. 處理姓名內的稱謂
DF['Title'] = DF['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
all['Title'] = all['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
title_dict = {
                "Mr" :        "Mr",
                "Miss" :      "Miss",
                "Mrs" :       "Mrs",
                "Master" :    "Master",
                "Dr":         "Scholar",
                "Rev":        "Religious",
                "Col":        "Officer",
                "Major":      "Officer",
                "Mlle":       "Miss",
                "Don":        "Noble",
                "Dona":        "Noble",
                "the Countess":"Noble",
                "Ms":         "Mrs",
                "Mme":        "Mrs",
                "Capt":       "Noble",
                "Lady" :      "Noble",
                "Sir" :       "Noble",
                "Jonkheer":   "Noble"
            }

DF['TitleGroup'] = DF['Title'].map(title_dict)
all['TitleGroup'] = all['Title'].map(title_dict)

# extracted title using name
all['Title'] = all.Name.str.extract(r'([A-Za-z]+)\.', expand=False)
all['Title'] = all['Title'].replace(['Capt', 'Col', 'Countess', 'Don',
                                               'Dr', 'Dona', 'Jonkheer', 
                                                'Major','Rev','Sir'],'Rare') 
all['Title'] = all['Title'].replace(['Mlle', 'Ms','Mme'],'Miss')
all['Title'] = all['Title'].replace(['Lady'],'Mrs')
all['Title'] = all['Title'].map({"Mr":0, "Rare" : 1, "Master" : 2,"Miss" : 3, "Mrs" : 4 })
Ti = all.groupby('Title')['Age'].median()
Ti_pred = all.groupby('Title')['Age'].median().values

DF['Ti_Age'] = DF['Age']
DF['Title'] = all.iloc[:len(DF)]['Title']

# 處理空值(age)
for i in range(0,5):
    DF.loc[(DF.Age.isnull()) & (DF.Title == i),'Ti_Age'] = Ti_pred[i]
train_feature['Ti_Age'] = DF['Ti_Age'].astype('int')
# 是否小於16歲(小孩)
train_feature['Ti_Minor'] = ((DF['Ti_Age']) < 16.0) * 1

# 2. 計算家人數目
DF['FamilySize'] =  DF['SibSp'] + DF['Parch']
DF.loc[DF['FamilySize'] == 0, 'Family'] = 'alone'
DF.loc[(DF['FamilySize'] > 0) & (DF['FamilySize'] <= 3), 'Family'] = 'small'
DF.loc[(DF['FamilySize'] > 3) & (DF['FamilySize'] <= 6), 'Family'] = 'medium'
DF.loc[DF['FamilySize'] > 6, 'Family'] = 'large'
# train_feature['Family'] = DF['Family'].map({'alone': 0, 'small': 1, 'medium': 2, 'large': 3}).astype(int)

# 3. 由於票價的全距相當大，故將所有數據依照5個區間分區
DF['Fare_5'] = pd.qcut(DF['Fare'], 5)

# 4. 有相同票的人
all['Connected_Survival'] = 0.5
for _, df_grp in all.groupby('Ticket'):
    if (len(df_grp) > 1):
        for ind, row in df_grp.iterrows():
            smax = df_grp.drop(ind)['Survived'].max()
            smin = df_grp.drop(ind)['Survived'].min()
            passID = row['PassengerId']
            if (smax == 1.0):
                all.loc[all['PassengerId'] == passID, 'Connected_Survival'] = 1
            elif (smin==0.0):
                all.loc[all['PassengerId'] == passID, 'Connected_Survival'] = 0

train_feature['Connected_Survival'] = all.iloc[:len(DF)]['Connected_Survival']

# 處理categorical data，one hot encoding
# train_feature['Embarked'] = train_feature['Embarked'].map({'C': 0, 'Q': 1, 'S':2}).astype(int)
train_feature['Sex'] = train_feature['Sex'].map({'male': 0, 'female': 1}).astype(int)
# train_feature = pd.concat([train_feature, pd.get_dummies(train_feature['Family'])], axis=1)

le = LabelEncoder()
le.fit(all['TitleGroup'])
# train_feature['TitleGroup'] = le.transform(DF['TitleGroup'])
train_feature['Fare_5'] = le.fit_transform(DF['Fare_5'])

# 做Normalization
scaler = StandardScaler()
# train_feature['Age'] = scaler.fit_transform(train_feature[['Age']])
train_feature['Ti_Age'] = scaler.fit_transform(train_feature[['Ti_Age']])

# 使用隨機森林來當模型
Model = RandomForestClassifier(random_state=2,n_estimators=250,min_samples_split=20,oob_score=True)
Model.fit(train_feature, train_labels)

mlp = MLPClassifier(solver='sgd', activation='relu',alpha=1e-4,hidden_layer_sizes=(50,50), random_state=1,max_iter=10,verbose=10,learning_rate_init=.1)
mlp.fit(train_feature, train_labels)

# 使用其他模型

# oob採用未被選用的data來做validation
print('Base oob score :%.5f' %(Model.oob_score_))

# 處理測試資料
# fare有空值
test.loc[ (test['Fare'].isnull()), 'Fare'] = test['Fare'].dropna().median()
# 選feature
test_feature = test[['Sex', 'Pclass']].copy()
# test_feature = test[['Age', 'Sex', 'Embarked', 'Pclass', 'SibSp', 'Parch']].copy()
# 新特徵
test['FamilySize'] = test['SibSp'] + test['Parch']
test.loc[test['FamilySize'] == 0, 'Family'] = 'alone'
test.loc[(test['FamilySize'] > 0) & (test['FamilySize'] <= 3), 'Family'] = 'small'
test.loc[(test['FamilySize'] > 3) & (test['FamilySize'] <= 6), 'Family'] = 'medium'
test.loc[test['FamilySize'] > 6, 'Family'] = 'large'

test['Title'] = test['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
test['TitleGroup'] = test['Title'].map(title_dict)
test.loc[ (test['TitleGroup'].isnull()), 'TitleGroup'] = test['TitleGroup'].dropna().mode() 
# 處理空值
test['Ti_Age'] = test['Age']
test['Title'] = all.iloc[len(DF):]['Title']

# Filling the missing age
for i in range(0,5):
    test.loc[(test.Age.isnull()) & (test.Title == i),'Ti_Age'] = Ti_pred[i]
test_feature['Ti_Age'] = test['Ti_Age'].astype('int')
# 是否小於16歲(小孩)
test_feature['Ti_Minor'] = ((test['Ti_Age']) < 16.0) * 1

test_feature['Connected_Survival'] = all.iloc[len(DF):]['Connected_Survival']

#one hot
# test_feature['Embarked'] = test_feature['Embarked'].fillna(test_feature['Embarked'].mode()[0])
# test_feature['Embarked'] = test_feature['Embarked'].map({'C': 0, 'Q': 1, 'S':2}).astype(int)
test_feature['Sex'] = test_feature['Sex'].map({'male': 0, 'female': 1}).astype(int)
# test_feature['Family'] = test['Family'].map({'alone': 0, 'small': 1, 'medium': 2, 'large': 3}).astype(int)
# test_feature = pd.concat([test_feature, pd.get_dummies(test_feature['Family'])], axis=1)

test['Fare_5'] = pd.qcut(test['Fare'], 5)
le_fare2 = LabelEncoder()
test_feature['Fare_5'] = le_fare2.fit_transform(test['Fare_5'])
test_feature[['Ti_Age']] = scaler.transform(test_feature[['Ti_Age']])
le.fit(all['TitleGroup'])
# test_feature['TitleGroup'] = le.transform(test['TitleGroup'])

# 預測
y_pred = Model.predict(test_feature)
y_pred_mlp = mlp.predict(test_feature)

from sklearn.metrics import accuracy_score
answer = pd.read_csv('https://raw.githubusercontent.com/johnny12150/yzu_cs_ml_courses/master/CS657%20ML/hw01%20titanic%20PLA/submission.csv')

print(accuracy_score(answer['Survived'], y_pred))

output = pd.DataFrame(
    {'PassengerId': test['PassengerId'],
     'Survived': y_pred_mlp
    })

# 輸出
output.to_csv('submit_RFtree.csv', index=False)

