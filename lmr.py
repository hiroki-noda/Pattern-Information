import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
features = np.loadtxt("data/auto_mpg.csv",skiprows = 0,usecols = (3,4,5),dtype = None)
print(features)
targets = np.loadtxt("data/auto_mpg.csv",skiprows = 0,usecols = (0),dtype = None)
a = np.ones(len(targets))
features = np.c_[features,a] #一列追加する

def w(X,t): #重みベクトルの計算
    return np.dot(np.linalg.pinv(X),t)

def main(x,y,z,w):
    print(w)
    return w[0]*x + w[1]*y + w[2]*z + w[3]

#グラフの描画
fig = plt.figure()
ax = Axes3D(fig)
x_ = features[:,0]
x = np.arange(0,400,80)
y_ = features[:,1]
y = np.arange(0,8000,1600)
z_ = features[:,2]
z = np.arange(0,30,6)
X,Y = np.meshgrid(x,y)
a = main(X,Y,z,w(features,targets))
ax.plot_wireframe(X,Y,a,color='green')
ax.scatter(x_,y_,targets)
ax.set_xlabel("horsepower")
ax.set_ylabel("weight")
ax.set_zlabel("mpg")
plt.show()
