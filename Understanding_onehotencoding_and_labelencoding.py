import pandas as pd
import numpy as np

df = pd.read_csv('onehotencoding_labelencoding.csv',delimiter = ',')

df.info()
df.describe()

# Import label encoder
from sklearn import preprocessing
# label_encoder object knows how to understand word labels
label_encoder= preprocessing.LabelEncoder()
# Encode labels in column 'Country'
df['Country']=label_encoder.fit_transform(df['Country'])
print(df.head())

# Importing one hot encoder

from sklearn.preprocessing import OneHotEncoder
# Creating one hot encoder object

onehotencoder = OneHotEncoder()
# Reshape the 1-D country array to 2-D as fit_transform expects 2-D and finally fit the object

X = onehotencoder.fit_transform(df['Country'].values.reshape(-1,1)).toarray()
donehot = pd.DataFrame(X,columns=[str(int(i)) for i in range(X.shape[1])])

df = pd.concat([donehot,df],axis=1)
df = df.drop('Country',axis=1)

# Dummy variable trap observed in the dataframe which is so-called Multicollinearity
# Multicollinearity occurs when there is a dependency between the independent variables
# which is a serious issue for the machine learning such as regression and classification
# to quantitatively check for the multicollinearity, the Variance Inflation Factor (VIF)
# could be implemented
# VIF = 1, very less multicollinearity
# VIF < 5, moderate multicollinearity
# VIF > 5, extreme multicollinearity needed to avoid

# Function to calculate VIF
import statsmodels.api as sm
def calculate_vif(data):
    vif_df = pd.DataFrame(columns = ['Var','Vif'])
    x_var_names = data.columns
    for i in range(0,x_var_names.shape[0]):
        y = data[x_var_names[i]]
        x = data[x_var_names.drop([x_var_names[i]])]
        r_squared = sm.OLS(y,x).fit().rsquared
        vif = round(1/(1-r_squared),2)
        vif_df.loc[i] = [x_var_names[i],vif]
    return vif_df.sort_values(by='Vif',axis=0,ascending=False,inplace=False)
X = df.drop(['Salary','0'],axis=1) # Salary column is a dependent col
calculate_vif(X)                    




 

