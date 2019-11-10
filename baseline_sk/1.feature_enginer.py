import pandas as pd
import numpy as np
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
from sklearn.preprocessing import OneHotEncoder
import time

start = time.clock()
df = pd.read_excel('./data/数据集.xlsx',None)
sheet_names = ['Train_DataSet','Test_DataSet','Train_DataSet_Label','Test_submit_example']
data_list = [0] * 4
for k in df.keys():
    data_list[sheet_names.index(k)] = df[k]
#1.查看训练集正负样本比例
print(data_list[2]['label'].value_counts())
'''
1    3659
2    2932
0     764
Name: label, dtype: int64
'''
X_train_df = data_list[0]
X_test_df = data_list[1]
y_train_df = data_list[2]
y_test_df = data_list[3]
train_drop_ids = [id for id in X_train_df['id'] if len(id) != 32]
X_train_df = X_train_df.drop(X_train_df[X_train_df.id.isin(train_drop_ids)].index)
test_drop_ids = [id for id in X_test_df['id'] if len(id) != 32]
X_test_df = X_test_df.drop(X_test_df[X_test_df.id.isin(test_drop_ids)].index)
X_train_merge = pd.merge(X_train_df,y_train_df,on='id')
X_test_merge = pd.merge(X_test_df,y_test_df,on='id')


X_train_title = X_train_merge['title']
X_train_content = X_train_merge['content']
y_train = X_train_merge['label']
X_test_title = X_test_merge['title']
X_test_content = X_test_merge['content']
y_test = X_test_merge['label']
ohe = OneHotEncoder()
y_train = ohe.fit_transform(y_train.values.reshape(-1,1))
y_test = ohe.transform(y_test.values.reshape(-1,1))

X_train_title_fenci_str_list = []
X_train_content_fenci_str_list = []
X_test_title_fenci_str_list = []
X_test_content_fenci_str_list = []
def fenci_str(text_list,fenci_str_list):
    for line in text_list:
        # if isinstance(line,float):
        if not isinstance(line,str):
            tmp = ' '
        else:
            tmp = ' '.join(jieba.lcut(line, cut_all=False))
        if tmp.endswith(' '):
            fenci_str_list.append(tmp)
        else:
            fenci_str_list.append(tmp + ' ')
fenci_str(X_train_title,X_train_title_fenci_str_list)
fenci_str(X_train_content,X_train_content_fenci_str_list)
fenci_str(X_test_title,X_test_title_fenci_str_list)
fenci_str(X_test_content,X_test_content_fenci_str_list)

X_train_text = []
X_test_text = []
for i in range(len(X_train_title_fenci_str_list)):
    X_train_text.append(X_train_title_fenci_str_list[i] + X_train_content_fenci_str_list[i])
for i in range(len(X_test_title_fenci_str_list)):
    X_test_text.append(X_test_title_fenci_str_list[i] + X_test_content_fenci_str_list[i])

# tfidf = TfidfVectorizer(ngram_range=(1,3))
# X_train_tfidf_vec = tfidf.fit_transform(X_train_text)
# X_test_tfidf_vec = tfidf.transform(X_test_text)
#
# joblib.dump(X_train_tfidf_vec,'./data/X_train_tfidf_vec')
# joblib.dump(X_test_tfidf_vec,'./data/X_test_tfidf_vec')
# joblib.dump(y_train,'./data/y_train_one_hot')
# joblib.dump(y_test,'./data/y_test_one_hot')
end = time.clock()
print('feature enginer time cost:{}'.format(end - start))

