import re
import jieba
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score

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
# tfidf分类
vectorizer = TfidfVectorizer()
train_tfidf = vectorizer.fit_transform(train_df['article']).toarray()
vocabulary = vectorizer.vocabulary_
val_tfidf = vectorizer.transform(val_df['article']).toarray()

models = {'xgb':xgb.XGBClassifier(max_depth=6,
                          learning_rate=0.1,
                          n_estimators=100,
                          objective="binary:logistic",
                          scale_pos_weight=10),
        "RidgeClassifier":RidgeClassifier(tol=1e-2, solver="sparse_cg"),
        "LogisticRegression":LogisticRegression(penalty='l2', random_state=0),
        "MultinomialNB": MultinomialNB(alpha=0.01),
        "RandomForestRegressor":RandomForestClassifier()
}


def validate_models(models, label_idx=4):
    """
    测试多个模型在某类标签上的模型表现
    :dict models: 多种机器学习模型的实例字典。
    :param label_idx: 选定的某一类标签，2-16，默认为4。因为4的分布最均衡。
    """
    print(f"{label_idx} \t {idx2label[label_idx]}")
    for name in models:
        model = models[name]
        train_Y = train_df['labels'].apply(lambda x: 1 if label_idx in x else 0)
        y_true = val_df['labels'].apply(lambda x: 1 if label_idx in x else 0)
        model.fit(train_tfidf, train_Y)
        y_pred = model.predict(val_tfidf)

        # for i, y in enumerate(y_pred):
        #     if y == 1:
        #         results[i].append(label_idx)
        # confusion_mat = metrics.confusion_matrix(val_Y,y_pred)
        print(f'model: \t {name}')
        print(f"accuracy_score: {accuracy_score(y_pred, y_true): .2f}")
        print(f"recall_score: {recall_score(y_true, y_pred): .2f}")
        print(f"precision_score: {precision_score(y_true, y_pred): .2f}")
        print("pred:", list(y_pred))
        print("true:", list(y_true))
        print("="*40)

def get_scores(models):
    """
    打印用多个模型进行分类的最后得分，每组预测值和结果完全一致才算正确。
    :dict models: 多种机器学习模型的实例字典。
    """
    for name in models:
        results = [[] for _ in range(len(val_df))]
        for label_idx in range(2, len(label2idx)): # [0,1] 为空集
            model = models[name]
            train_Y = train_df['labels'].apply(lambda x: 1 if label_idx in x else 0)
            y_true = val_df['labels'].apply(lambda x: 1 if label_idx in x else 0)
            model.fit(train_tfidf, train_Y)
            y_pred = model.predict(val_tfidf)

            for i, y in enumerate(y_pred):
                if y == 1:
                    results[i].append(label_idx)

        score = 0
        for i, row in enumerate(results):
            pred = results[i]
            true = val_df.iloc[i]['labels']
            if set(true) == set(pred):
                print(true, pred)
                score += 1
        print(f"Score of {name}: {score}")

get_scores(models)


