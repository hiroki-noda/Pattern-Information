import numpy as np
import matplotlib.pyplot as plt
import random
features = np.loadtxt("data/iris.csv",delimiter = ",", skiprows=0, usecols = (0,1,2,3))

def distance(p,q):
    return np.linalg.norm(p-q)

def predict(test,rep):
    distances = []
    for p in rep:
        distances.append(distance(p,test)) #各代表点からの距離を格納
    Near_indexes = np.argsort(distances) #近い順にソートし、インデックスを得る
    Nearest_index = Near_indexes[0] #最も近い代表点のインデックスを得る
    return Nearest_index

def rep_point(features,predicted_labels,clusters): #重心を求める
    rep_points = [] #代表点の集合
    for i in range(clusters): #クラスタの数だけ計算
        label_i = np.zeros(4) #i番目のラベルの特徴点の和
        counter_i = 0 #i番目のラベルの特徴点数
        for j in range(len(features)): #すべての特徴点を調べる
            if predicted_labels[j] == i:
                label_i += features[j]
                counter_i += 1
        rep_i = label_i/counter_i #i番目のラベルの重心を計算
        rep_points.append(rep_i)
    return rep_points


def main(features,clusters):
    predicted_labels = []
    for i in range(len(features)):
        predicted_labels.append(i % clusters) #ひとまず適当にラベルを割り振る
    init_rep = np.zeros((clusters,4))
    for i in range(clusters):
        init_rep[i] = features[random.randint(0,len(features)-1)] #初期代表点を特徴点の中から適当に選ぶ
    rep = init_rep
    for n in range(20):
        for i in range(len(features)):
            predicted_labels[i] = predict(features[i],rep) #ラベルを更新
            rep = rep_point(features,predicted_labels,clusters) #代表点を更新

#グラフの描画
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    colors = ['red','blue','yellow','pink','green']
    for i in range(clusters):
        x = np.array([])
        y = np.array([])
        for j in range(len(features)):
            if predicted_labels[j] == i:
                x = np.append(x,features[j,0])
                y = np.append(y,features[j,2])
        ax.scatter(x,y,c = colors[i])
    for i in range(clusters):
        ax.scatter(rep[i][0], rep[i][2], s=100,facecolors='none', edgecolors='black')
    plt.show()

for k in range(2,6):
    main(features,k)
