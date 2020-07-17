import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
targets = np.loadtxt("data/iris.csv",delimiter = ",", skiprows=0, usecols = 4, dtype = "unicode")
features = np.loadtxt("data/iris.csv",delimiter = ",", skiprows=0, usecols = (0,1,2,3))
def distance(p,q):
    return np.linalg.norm(p-q)

def predict(train_data,train_labels,k,test_data):
    distances = []
    for p in train_data:
        distances.append(distance(p,test_data)) #各training dataとの距離を計算し、配列に追加
    Near_indexes = np.argsort(distances) #近い順にソートし、インデックスを得る
    KNearest_indexes = Near_indexes[:k] #k番目までのインデックスを取り出す
    KNearest_labels = train_labels[KNearest_indexes] #k番目までのラベルを取り出す
    c = Counter(KNearest_labels)
    return c.most_common(1)[0][0] #最も多いラベルを予測ラベルとする

def main(features,targets,k):
    counter = 0
    for i in range(len(features)):
        train_data = features[1:150]
        train_labels = targets[1:150]
        correct_label = targets[0]
        predicted_label = predict(train_data,train_labels,k,features[0])
        if predicted_label == correct_label:
            counter += 1
        #一つずらす
        features = np.roll(features,-1,axis = 0)
        targets = np.roll(targets,-1,axis = 0)
    N = len(features)
    print('score = {0}'.format(counter/N)) #正答率を計算
    return counter/N

#グラフの描画
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
x = np.array([])
y = np.array([])
for k in range(1,30):
    x = np.append(x,k)
    y = np.append(y,main(features,targets,k))
ax.plot(x,y)
plt.xlabel("k")
plt.ylabel("accuracy")

plt.show()
