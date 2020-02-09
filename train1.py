import sys
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras

numbers = ['0', '2', '3', '4', '5', '6', '7', '8', '9', '10']
alphbets = ['A', 'J', 'Q', 'K']
shape = ['hongtao', 'fangpian', 'heitao', 'meihua']
other = ['other']
dataset = numbers + alphbets + shape + other
dataset_len = len(dataset)

num_epochs = 50
batch_size = 100
learning_rate = 0.001

cur_dir = sys.path[0]
train_data_dir = os.path.join(cur_dir, 'pic/train')
train_model_path = os.path.join(cur_dir, 'model/char_recongnize/model1.ckpt')
test_data_dir = os.path.join(cur_dir, 'pic/test')

def list_all_files(root):
    files = []
    list = os.listdir(root)
    for i in range(len(list)):
        element = os.path.join(root, list[i])
        if list[i] == '.DS_Store':
            continue
        if os.path.isdir(element):
            temp_dir = os.path.split(element)[-1]
            if temp_dir in dataset:
                files.extend(list_all_files(element))
        elif os.path.isfile(element):
            files.append(element)
    return files


def init_data(dir):
    X = []
    y = []
    if not os.path.exists(train_data_dir):
        raise ValueError('没有找到文件夹')
    files = list_all_files(dir)

    for file in files:
        src_img = cv2.imread(file, cv2.COLOR_BGR2GRAY)
        if src_img.ndim == 3:
            continue
        resize_img = cv2.resize(src_img, (20, 20))
        X.append(resize_img)
        # 获取图片文件全目录
        dir = os.path.dirname(file)
        # 获取图片文件上一级目录名
        dir_name = os.path.split(dir)[-1]
        # vector_y = [0 for i in range(len(dataset))]
        index_y = dataset.index(dir_name)
        # vector_y[index_y] = 1
        y.append(index_y)

    X = np.array(X)
    X = X.reshape(-1,20,20,1)

    y = np.array(y)
    return X, y

def init_testData(dir):
    test_X = []
    if not os.path.exists(dir):
        raise ValueError('没有找到文件夹')
    files = list_all_files(dir)
    for file in files:
        src_img = cv2.imread(file, cv2.COLOR_BGR2GRAY)
        if src_img.ndim == 3:
            continue
        resize_img = cv2.resize(src_img, (20, 20))
        test_X.append(resize_img)
    test_X = np.array(test_X)
    test_X = test_X.reshape(-1, 20, 20, 1)
    test_X = test_X.astype('float32')
    return test_X

def train():
    # 加载训练集
    X, y = init_data(train_data_dir)

    train_dataset = tf.data.Dataset.from_tensor_slices((X, y))
    # 取出前buffer_size个数据放入buffer，并从其中随机采样，采样后的数据用后续数据替换
    train_dataset = train_dataset.shuffle(buffer_size=23000)
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    # 构建堆叠网络模型
    rate = 0
    model = tf.keras.Sequential([
        # layers.Reshape(target_shape=(-1, 20, 20, 1),input_shape=[20, 20, 1]),
        layers.Conv2D(filters=32,  # 卷积层神经元（卷积核）数目
                      kernel_size=3,  # 感受野大小
                      strides=1,  # 移动补偿
                      padding='same',  # padding策略（vaild 或 same）
                      activation=tf.nn.relu,  # 激活函数
                      input_shape=[20, 20, 1]
                      ),
        # layers.Activation('relu'),
        layers.MaxPool2D(pool_size=2, padding='same'),
        layers.Dropout(rate),
        layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu),
        layers.MaxPool2D(pool_size=2, padding='same'),
        layers.Dropout(rate),
        layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu),
        layers.MaxPool2D(pool_size=2, padding='same'),
        layers.Dropout(rate),
        layers.Reshape(target_shape=(3 * 3 * 128,)),
        # layers.Dense(units=3 * 3 * 128, activation=tf.nn.relu),
        # layers.Dropout(rate),
        layers.Dense(units=1024, activation=tf.nn.relu),
        layers.Dropout(rate),
        layers.Dense(units=dataset_len),
        layers.Softmax()
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), # 归一化器
        loss=tf.keras.losses.sparse_categorical_crossentropy, # loss计算方法
        metrics=[tf.keras.metrics.sparse_categorical_accuracy]  # acc计算方法
    )
    model.summary()
    callbacks = [
        keras.callbacks.ModelCheckpoint(  # 设置自动保存模型和acc周期回显
            train_model_path,
            monitor='val_acc',  # 监控acc属性
            verbose=1,  # 显示结果
            save_best_only=True,  # 仅保存最好的模型
            save_weights_only=False,
            mode='auto',
            period=10  # 检查间隔
        )
    ]
    model.fit(train_dataset, epochs=num_epochs, callbacks=callbacks)
    model.save(train_model_path)

def test():
    model = tf.saved_model.load(train_model_path)
    X = init_testData(test_data_dir)
    y_pred = model(X)
    print(y_pred)
    for y in y_pred:
        print(dataset[np.argmax(y)])
    # print(dataset[y_pred])
    sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    # num_batches = int(len(X) // batch_size)
    # for batch_index in range(num_batches):
    #     start_index, end_index = batch_index * batch_size, (batch_index + 1) * batch_size
    #     y_pred = model(X[start_index: end_index])
    #     print(X)
    #     print(dataset[y_pred])
        # sparse_categorical_accuracy.update_state(y_true=X[start_index: end_index], y_pred=y_pred)
    # print("test accuracy: %f" % sparse_categorical_accuracy.result())

if __name__ == '__main__':
    # train()
    test()