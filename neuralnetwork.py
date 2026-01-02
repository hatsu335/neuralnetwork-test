import numpy as np
import gzip
import struct
import os
import time
from typing import cast    # VSCodeでの警告対策(castは無くても問題なく起動する)

# ユーティリティ関数

def normalize(record):
    return record / 255.0 * 0.99 + 0.01

def one_hot(label, size=10):
    vec = np.zeros(size) * 0.01
    vec[int(label.item())] = 0.99
    return vec

# データ読み込み関数

def inputs_file(filename):
    with gzip.open(filename, 'rb') as f:
        _magic, total_images, rows, columns = struct.unpack('>IIII', cast(bytes, f.read(16)))
        image_data = cast(bytes, f.read(total_images * rows * columns))
        images = np.frombuffer(image_data, dtype=np.uint8)
        return np.asarray(images.reshape(total_images, rows * columns))

def targets_file(filename):
    with gzip.open(filename, 'rb') as f:
        _magic, total_label = struct.unpack('>II', cast(bytes, f.read(8)))
        label_data = cast(bytes, f.read(total_label))
        labels = np.frombuffer(label_data, dtype=np.uint8)
        return np.asarray(labels)

# NeuralNetwork本体

class NeuralNetwork(object):
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate
        
        self.wih = np.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))
        
        self.hidden_activation = lambda x: np.maximum(0, x)  #ReLU
        self.hidden_derivation = lambda x: (x > 0).astype(float)  #微分
        
        self.output_activation = lambda x: 1 / (1 + np.exp(-x))  # Sigmoid
        self.output_derivation = lambda y: y * (1 - y)  #微分
    
    def train(self, inputs_list, targets_list, epoch):
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.hidden_activation(hidden_inputs)
        
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.output_activation(final_inputs)
        
        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors)
        
        self.who += self.lr ** (epoch + 1) * np.dot((output_errors * self.output_derivation(final_outputs)), hidden_outputs.T)
        self.wih += self.lr ** (epoch + 1) * np.dot((hidden_errors * self.hidden_derivation(hidden_inputs)), inputs.T)
        
    def query(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T
        
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.hidden_activation(hidden_inputs)
        
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.output_activation(final_inputs)
        
        return final_outputs
    
# 学習設定

input_nodes = 784
hidden_nodes = 200
output_nodes = 10
learning_rate = 0.1
# decreasing_rate = 1.0
epochs = 1

n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
start_time = time.time()

train_image_data = inputs_file('MNIST/train-images-idx3-ubyte.gz')
train_target_data = targets_file('MNIST/train-labels-idx1-ubyte.gz')
test_image_data = inputs_file('MNIST/t10k-images-idx3-ubyte.gz')
test_target_data = targets_file('MNIST/t10k-labels-idx1-ubyte.gz')

# 保存フォルダ準備

os.makedirs('models', exist_ok=True)

# 学習ループ

for e in range(epochs):
    print(f"Epoch {e+1}")
    for i, record in enumerate(train_image_data):
        inputs = normalize(record)
        targets = one_hot(train_target_data[i])
        n.train(inputs, targets, e)
    
    # 重み保存
    
    np.save(f"models/wih_epoch{e+1}.npy", n.wih)
    np.save(f"models/who_epoch{e+1}.npy", n.who)

    # 評価
    
    correct = 0
    for i, record in enumerate(test_image_data):
        inputs = normalize(record)
        label = int(test_target_data[i])
        output = n.query(inputs)
        prediction = np.argmax(output)
        if prediction == label:
            correct += 1
            
    performance = correct / len(test_image_data)
    print(f"Performance  : {performance:.4f}")
end_time = time.time()
print(f"hidden nodes   : {hidden_nodes}")
print(f"learning rate  : {learning_rate}")
# print(f"decreasing rate: {decreasing_rate}")
print(f"epochs         : {epochs}")
print(f"time           : {end_time - start_time:.2f}")
