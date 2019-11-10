import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import numpy as np
#构建字典
data_path = './data/数据集.xlsx'
vocab_path = './data/vocab.txt'
def build_vocab(file_path,vocab_path):
    X_train_title,X_train_content,_ = read_file(file_path)
    train_text = X_train_title + X_train_content
    word_set = set(''.join(train_text))
    word_dict = dict(zip(word_set,range(1,len(word_set)+1)))
    word_dict['<PAD>'] = 0
    f = open(vocab_path,'w',encoding='utf-8')
    f.write(str(word_dict))
    f.close()
    print('字典大小:{}'.format(len(word_dict)))


def read_file(file_path):
    df = pd.read_excel(file_path, None)
    sheet_names = ['Train_DataSet', 'Test_DataSet', 'Train_DataSet_Label', 'Test_submit_example']
    data_list = [0] * 4
    for k in df.keys():
        data_list[sheet_names.index(k)] = df[k]
    # 1.查看训练集正负样本比例
    print(data_list[2]['label'].value_counts())
    '''
        1    3659
        2    2932
        0     764
        Name: label, dtype: int64
        '''
    X_train_df = data_list[0]
    y_train_df = data_list[2]
    train_drop_ids = [id for id in X_train_df['id'] if len(id) != 32]
    X_train_df = X_train_df.drop(X_train_df[X_train_df.id.isin(train_drop_ids)].index)
    X_train_merge = pd.merge(X_train_df, y_train_df, on='id')
    X_train_title = X_train_merge['title']
    X_train_content = X_train_merge['content']
    print('X_train_content:{},X_train_title:{},y_train:{}'.format(len(X_train_content),len(X_train_title),len(X_train_merge['label'])))
    return X_train_title.astype(str),X_train_content.astype(str),X_train_merge['label']


def read_vocab(file_path):
    f = open(file_path,'r',encoding='utf-8')
    vocab = eval(f.read())
    print('字典大小:{}'.format(len(vocab)))
    f.close()
    return vocab

def process_file(file_path,vocab_path,max_length):
    #1.输出[全部文本,前一百个字的id]
    #2.输出[全部文本,标签]
    #先把文本转成id,然后不够100的补0,多的截取100
    X_train_title,X_train_content,y_train = read_file(file_path)
    print('X_train_title:{}'.format(len(X_train_title)))
    print('X_train_content:{}'.format(len(X_train_content)))
    train_text = X_train_title + X_train_content
    print('train_text:{}'.format(len(train_text)))
    word_2_id = read_vocab(vocab_path)
    train_id = []
    for text in train_text:
        tmp = []
        for w in text:
            try:
                tmp.append(word_2_id[w])
            except:
                print(w)
                tmp.append(0)
        train_id.append(tmp)
    # print('train_id:{}'.format(np.array(train_id).shape))
    print(train_id[:5])
    print('train_id:{}'.format(len(train_id)))
    train_pad = []
    for ids in train_id:
        if len(ids) >= max_length:
            train_pad.append(ids[:100])
        else:
            ids.extend([0] * (100-len(ids)))
            train_pad.append(ids)
    print('train_pad:{}'.format(len(train_pad)))
    print('train_pad:{}'.format(train_pad[:2]))
    print('train_pad:{}'.format(np.array(train_pad).shape))
    one_hot = OneHotEncoder()
    y_train = one_hot.fit_transform(y_train.reshape(-1,1)).toarray()
    y_0 = []
    y_1 = []
    y_2 = []
    for i, arr in enumerate(y_train):
        if np.argmax(arr) == 0:
            y_0.append(i)
        elif np.argmax(arr) == 1:
            y_1.append(i)
        else:
            y_2.append(i)
    print(len(y_0))
    print(len(y_1))
    print(len(y_2))
    y_1_train = y_1[:763]
    y_1_test = y_1[763:]
    y_2_train = y_2[:763]
    y_2_test = y_2[763:]
    train_0 = [v for i, v in enumerate(train_pad) if i in y_0]
    print(np.array(train_0).shape)
    train_1 = [v for i, v in enumerate(train_pad) if i in y_1_train]
    train_2 = [v for i, v in enumerate(train_pad) if i in y_2_train]
    test_1 = [v for i, v in enumerate(train_pad) if i in y_1_test]
    test_2 = [v for i, v in enumerate(train_pad) if i in y_2_test]
    x_train = train_0 + train_1 + train_2
    y_train_res = [v for k,v in enumerate(y_train) if k in y_0 + y_1_train + y_2_train]
    x_test = test_1 + test_2
    y_test = [v for k,v in enumerate(y_train) if k in y_1_test + y_2_test]
    # x_train,x_test,y_train,y_test = train_test_split(train_pad,y_train,test_size=0.2)
    print('训练集容量:{}'.format(len(x_train)))
    print('x_train:{}'.format(np.array(x_train).shape))
    print('x_test:{}'.format(np.array(x_test).shape))
    print('y_train:{}'.format(np.array(y_train_res).shape))
    print('y_test:{}'.format(np.array(y_test).shape))
    return x_train,x_test,y_train_res,y_test

# build_vocab(data_path,vocab_path)
process_file(data_path,vocab_path,100)