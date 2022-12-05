import pandas as pd
import jieba
import jieba

def get_dataframe(txt_file):
    data_dict = {'article':[], "labels":[]}

    with open(txt_file, 'r')as f:
        for line in f.readlines():
            labels, article = line.split('\t')
            data_dict['labels'].append(labels.strip().strip(','))
            data_dict['article'].append(article)
    return pd.DataFrame(data_dict)

train_df = get_dataframe('train.txt')
val_df = get_dataframe('val.txt')

# 将labels处理成数字组合
labels_series = train_df['labels'].str.split(',')
label_set = set()
for labels in labels_series:
    for label in labels:
        label_set.add(label)
label_set.add('')

label_df = pd.DataFrame(label_set, columns=['label'])

train_df.to_csv('./dataset/train.csv', index_label='id')
val_df.to_csv('./dataset/val.csv', index_label='id')
label_df.to_csv('./dataset/label.csv', index_label='id')