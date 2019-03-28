# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 22:32:38 2019

@author: Purandhar
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import keras

np.random.seed(2)
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
data = pd.read_csv(r'C:\Users\Purandhar\Desktop\Capstone\creditcard_forgit.csv', engine='python')
data.head()
data.info()
#computing correlation with Class vairable
data.corrwith(data.Class).plot.bar(
        figsize = (20, 10), title = "Correlation with class", fontsize = 15,
        rot = 45, grid = True)
## Correlation Matrix
sn.set(style="white")

# generate the correlation matrix
corr = data.corr()
corr.head()
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(18, 15))

# Generate a custom diverging colormap
cmap = sn.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sn.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
data.isna().any()
#scaling data 
from sklearn.preprocessing import StandardScaler
data['normalizedAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1,1))
data = data.drop(['Amount'],axis=1)
data = data.drop(['Time'],axis=1)
data.head()
X = data.iloc[:, data.columns != 'Class']
y = data.iloc[:, data.columns == 'Class']
X.info()
y.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state=0)
from yellowbrick.features import PCADecomposition
visualizer=PCADecomposition(proj_dim=3)
visualizer.fit_transform(X_test,y_test)
visualizer.poof()

#Model
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
#try some different activation function 
classifier.add(Dense(units =15 , kernel_initializer = 'uniform', activation = 'relu', input_dim = 29))

# Adding the second hidden layer
classifier.add(Dense(units = 15, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 32, epochs = 3)#reduce epochs accuracy doesn't improve that much
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
score = classifier.evaluate(X_test, y_test)
score

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm
#confusion matrix is not good metric for imbalanced data, use roc_auc_score(y_test,y_pred)  :AUROC score
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
#from yellowbrick.classifier import ClassificationReport
#visualizer=ClassificationReport(Sequential)
#visualizer.fit(X_train, y_train)
#visualizer.score(X_test, y_test)
#visualizer.poof()
#same output as earlier 2 commands but with a heat map
cm = confusion_matrix(y_test, y_pred) # rows = truth, cols = prediction
df_cm = pd.DataFrame(cm, index = (0, 1), columns = (0, 1))
plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)
sn.heatmap(df_cm, annot=True, fmt='g')
print("Test Data Accuracy: %0.4f" % accuracy_score(y_test, y_pred))
