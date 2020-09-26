from sklearn import datasets 
iris = datasets.load_iris()
import matplotlib.pyplot as plt
import numpy as np
from mlxtend.plotting import plot_decision_regions
sepal_length = iris.data[:,0]
sepal_width = iris.data[:,1]

plt.scatter(sepal_length,sepal_width,c=iris.target)
from sklearn.svm import SVC
# Kernel and C in SVM
clf = SVC(kernel = 'rbf',C=10)

clf.fit(np.c_[sepal_length,sepal_width],iris.target.reshape(150,1))

plot_decision_regions(np.c_[sepal_length,sepal_width],y=iris.target.astype(np.integer),clf=clf)

from sklearn.metrics.pairwise import sigmoid_kernel
sigmoid_kernel(np.c_[sepal_length,sepal_width])
from sklearn.preprocessing import normalize
sepal_length_norm = normalize(sepal_length.reshape(1,-1))[0]
sepal_width_norm = normalize(sepal_width.reshape(1,-1))[0]
feature = np.c_[sepal_length_norm,sepal_width_norm]
clf.fit(feature,iris.target)
plot_decision_regions(feature,iris.target,clf=clf)

# the effect on Gamma parameter on the performance of SVM
petal_length = iris.data[:,2]
clf = SVC(kernel='rbf',gamma=100)
feature = np.c_[sepal_length,petal_length]
clf.fit(feature,iris.target)
plot_decision_regions(feature,iris.target,clf=clf)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
scores = []
for i in range(0,500):
    X_train, X_test, y_train, y_test=\
    train_test_split(feature,iris.target,test_size=0.2)
    clf = SVC(kernel='rbf', C=10, gamma=0.001)
    clf.fit(X_train,y_train)
    scores.append(accuracy_score(clf.predict(X_test),y_test))
plt.hist(scores)    
1/(2*feature.var())
clf = SVC(kernel='rbf',C=10,gamma=(1/(2*feature.var())))
clf.fit(feature,iris.target)
plot_decision_regions(feature,iris.target,clf=clf)
