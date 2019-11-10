import tensorflow as tf
from RNN_feature_enginer import *
import numpy as np
#搭建基于RNN的中文文本分类模型
#1.定义需要的参数
#入参[batch_size,time_step,feature]
batch_size = 32
time_step = 100
feature = 128
#cell的隐层,output的维度
hidden_size = 128
#字典的容量
vocab_size = 7624
#最终分类
n_class = 3
#2.网络搭建
#2.1数据接收器:[batch_size,time_step]
input = tf.placeholder(tf.int32,[None,time_step])
label = tf.placeholder(tf.float32,[None,n_class])
#2.2 keep_pro
keep_pro = tf.placeholder(tf.float32)
#embedding,这个也是要训练的要定义一个变量
embedding = tf.get_variable('embedding',[vocab_size,feature])
#[batch_size,time_step,feature]
embedding_input = tf.nn.embedding_lookup(embedding,input)
#建立双层单向RNN网络
lstm_cell1 = tf.nn.rnn_cell.GRUCell(hidden_size)
lstm_cell1 = tf.nn.rnn_cell.DropoutWrapper(lstm_cell1,output_keep_prob=keep_pro)
lstm_cell2 = tf.nn.rnn_cell.GRUCell(hidden_size)
lstm_cell2 = tf.nn.rnn_cell.DropoutWrapper(lstm_cell2,output_keep_prob=keep_pro)
mul_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell1,lstm_cell2])
#获取输出output,lastState
output,lastState = tf.nn.dynamic_rnn(mul_cell,inputs=embedding_input,dtype=tf.float32)
#获取最后的状态输出[batch_size,hidden_size]
x = output[:,-1,:]
#创建分类器
fc = tf.layers.dense(x,128)
fc = tf.nn.relu(fc)
fc = tf.nn.dropout(fc,keep_pro)

logist = tf.layers.dense(fc,n_class)

loss = tf.reduce_mean(tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=label,logits=logist)))
train = tf.train.AdamOptimizer().minimize(loss)

predict = tf.argmax(tf.nn.softmax(logist),1)
y = tf.argmax(label,1)
cp = tf.equal(predict,y)
accuracy = tf.reduce_mean(tf.cast(cp,tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables())
    data_path = './data/数据集.xlsx'
    vocab_path = './data/vocab.txt'
    x_train, x_test, y_train, y_test = process_file(data_path, vocab_path, 100)
    batch = len(x_train) // batch_size
    for epoch in range(50):
        for batch_step in range(batch):
            start = batch_step * batch_size
            stop = start + batch_size
            batch_x = np.array(x_train[start:stop])
            batch_y = np.array(y_train[start:stop])
            # print('batch_x:{}'.format(batch_x.shape))
            # print('batch_y:{}'.format(batch_y.shape))
            _,train_loss = sess.run([train,loss],feed_dict={input:batch_x,label:batch_y,keep_pro:0.618})
            print('epoch:{}, batch:{},train_loss:{}'.format(epoch+1, batch_step+1, train_loss))
        train_accuray = sess.run(accuracy,feed_dict={input:x_train,label:y_train,keep_pro:1.0})
        test_accuracy = sess.run(accuracy,feed_dict={input:x_test,label:y_test,keep_pro:1.0})
        print('epoch:{},train_accuracy:{},test_accuracy:{}'.format(epoch+1,train_accuray,test_accuracy))
        if epoch%1==0:
            saver.save(sess,'./data/model',global_step=epoch)