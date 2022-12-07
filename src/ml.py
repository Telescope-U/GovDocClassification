import re
import jieba
import pandas as pd
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.metrics import accuracy_score, recall_score, precision_score
import xgboost as xgb

label_df = pd.read_csv('../dataset/label.csv')
train_df = pd.read_csv('../dataset/train.csv')
val_df = pd.read_csv('../dataset/val.csv')

label2idx = {line['label']:line['id'] for i, line in label_df.iterrows()}
idx2label = {line['id']: line['label'] for i, line in label_df.iterrows()}
# 将label处理成数字化
def trans_label2idx(str_list):
    return [label2idx[label] for label in str_list]
train_df['labels'] = train_df['labels'].str.split(',').apply(trans_label2idx)
val_df['labels'] = val_df['labels'].str.split(',').apply(trans_label2idx)

""" 中文分词 """
def split_cn(text, sep=' '):
    punc = "[’!'\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+，。！？“”《》：、． \n（）〕〔. "
    a = [word for word in jieba.lcut(text) if (word not in punc) and (re.search('[0-9]|x', word) == None)]
    return sep.join(a)

train_df['article'] = train_df['article'].apply(lambda x: split_cn(x))
val_df['article'] = val_df['article'].apply(lambda x: split_cn(x))

vectorizer = TfidfVectorizer()
train_tfidf = vectorizer.fit_transform(train_df['article']).toarray()
vocabulary = vectorizer.vocabulary_
val_tfidf = vectorizer.transform(val_df['article']).toarray()

results = [[] for _ in range(len(val_df))]

# for label_idx in range(2, len(label2idx)):
label_idx = 4
model = xgb.XGBClassifier(max_depth=6,
                          learning_rate=0.1,
                          n_estimators=100,
                          objective="binary:logistic",
                          scale_pos_weight=10)
train_Y = train_df['labels'].apply(lambda x: 1 if label_idx in x else 0)
y_true = val_df['labels'].apply(lambda x: 1 if label_idx in x else 0)
model.fit(train_tfidf, train_Y)
y_pred = model.predict(val_tfidf)


# for i, y in enumerate(y_pred):
#     if y == 1:
#         results[i].append(label_idx)
# confusion_mat = metrics.confusion_matrix(val_Y,y_pred)
print(f"{label_idx} \t {idx2label[label_idx]}")
print(f"accuracy_score: {accuracy_score(y_pred, y_true): .2f}")
print(f"recall_score: {recall_score(y_true, y_pred): .2f}")
print(f"precision_score: {precision_score(y_true, y_pred): .2f}")
print("pred:", list(y_pred))
print("true:", list(y_true))
print("="*150)
# i = 0
# score = 0
# for i, row in enumerate(results):
#     pred = results[i]
#     true = val_df.iloc[i]['labels']
#     if set(true) == set(pred):
#         score += 1
# print(score)


