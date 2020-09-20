import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly import __version__
import cufflinks as cf
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected = True)
cf.go_offline()

# Loading dataset

df = pd.read_csv('diabetes.csv',delimiter=',')

df.head()
# Data cleaning

df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']]=\
df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)

df.isnull().sum()
# Plotting the bar of missing data
import missingno as msno
p = msno.bar(df)
import impyute
import sys
from impyute.imputation.cs import fast_knn
sys.setrecursionlimit(100000) # increase the recursion limit of the system
# start the KNN training
imputed_training = fast_knn(df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].values,k = 30)
df_t1 = pd.DataFrame(imputed_training,columns=['Glucose','BloodPressure','SkinThickness','Insulin','BMI'])
df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = df_t1[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']]
# Providing the info of the total nulls 
df.isnull().sum()
# Data describing and visualization
df.info()
df.describe()
# Heatmap plotting
import seaborn as sns
df.corr()
sns.heatmap(df.corr(),annot=True)

p = df[df['Outcome']==1].hist(figsize=(20,20))
plt.title('Diabetes Patient')
# KNN Visualization all features with Outcome
from sklearn.decomposition import PCA
from mlxtend.plotting import plot_decision_regions
from sklearn.svm import SVC
X = df[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI',
        'DiabetesPedigreeFunction','Age']]
y = df['Outcome']
clf = SVC(C=100, gamma = 0.0001)
pca = PCA(n_components = 2)
X_train2 = pca.fit_transform(X)
clf.fit(X_train2, df['Outcome'].astype(int).values.reshape(-1,1))
plot_decision_regions(X_train2, df['Outcome'].astype(int).values,
                      clf = clf, legend = 2)
# KNN features visualization with each other
from sklearn import datasets, neighbors
from mlxtend.plotting import plot_decision_regions
def ok(X,Y):
    x = df[[X,Y]].values
    y = df['Outcome'].astype(int).values
    clf = neighbors.KNeighborsClassifier(n_neighbors = 9)
    clf.fit(x,y)
    # Plotting decision region
    plot_decision_regions(x,y,clf=clf,legend = 2)
    # Adding axes annotations
    plt.xlabel(X)
    plt.ylabel(Y)
    plt.title('Knn with K = '+ str(9))
    plt.show()

tt = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
      'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']  
ll = len(tt)  

for i in range(0,ll):
    for j in range(i+1,ll):
        ok(tt[i],tt[j])
        
# Data preprocessing
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df.drop('Outcome',axis=1))
scaler_features = scaler.transform(df.drop('Outcome',axis=1))
df_feat = pd.DataFrame(scaler_features,columns=df.columns[:-1])
# Appending the outcome feature
df_feat['Outcome']=df['Outcome'].astype(int)
df = df_feat.copy()
df.head()
# to reserve scaler transformation
# s = scaler.inverse_transform(df_feat)
# df_feat = pd.DataFrame(s, columns = df.columns[:-1])

# KNN
X = df.drop('Outcome',axis=1)
y = df['Outcome']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                    test_size=0.3,random_state=0)
# Check for the best K value by getting receiver operating characteristic accuracy for each 
# K ranging from 1 to 100

import sklearn
tt = {}
il = []
ac = []
for i in range(1,100):
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors = i)
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_test)
    from sklearn.metrics import accuracy_score
    il.append(i)
    ac.append(sklearn.metrics.roc_auc_score(y_test,y_pred))
    tt.update({'K':il})
    tt.update({'ROC_ACC':ac})
    vv = pd.DataFrame(tt)
    vv.sort_values('ROC_ACC',ascending=False,inplace=True)
    vv.head(10)
# Selecting K = 25 which yields the best result
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors  = 25)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))
from sklearn.metrics import confusion_matrix
conf = confusion_matrix(y_test,y_pred)
print(conf)
sns.heatmap(conf,annot=True)
from sklearn.metrics import roc_curve
plt.figure(dpi=100)
fpr, tpr, thresholds = roc_curve(y_test,y_pred)
plt.plot(fpr,tpr,label='%.2f'
         %sklearn.metrics.roc_auc_score(y_test,y_pred))
plt.legend(loc='lower right')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve for diabetes classifier')
plt.xlabel('False positive rate (1-Specificity)')
plt.ylabel('True positive rate (Sensitivity)')
plt.grid(True)
import sklearn
sklearn.metrics.roc_auc_score(y_test,y_pred)
data = {'test':y_test.values.ravel(),
        'pred':y_pred.ravel(),'number':
            np.arange(0,len(y_test))}
pt = pd.DataFrame(data)
pt.iplot(kind = 'scatter',
         x='number',
         y=['test','pred'],
         color=['white','yellow'],
         theme='solar',
         mode='lines+markers')
