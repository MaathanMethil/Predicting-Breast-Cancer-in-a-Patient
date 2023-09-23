#!/usr/bin/env python
# coding: utf-8

# # Predicting Breast Cancer in a patient

# Abstract:
#     Breast cancer represents one of the diseases that make a high number of deaths every
#     year. It is the most common type of all cancers and the main cause of women's deaths
#     worldwide. Classification and data mining methods are an effective way to classify data.
#     Especially in the medical field, where those methods are widely used in diagnosis and
#     analysis to make decisions.

# In[ ]:





# In[5]:


# importing the libraries:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
#
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


# ● Analysing the available data and exploring relationships among given variables ● 

# In[6]:


# importing the data:
pbc = pd.read_csv(r"C:\Users\DONMETHIL\Downloads\Xudemy\cancerdata.csv")


# In[7]:


pbc.head() # listing the dataframe


# In[ ]:





# In[8]:


pbc.columns # listing the columns


# In[11]:


pbc.shape # 569 rows and 33 columns


# In[12]:


pbc.info() # show the datatypes and null/non-null entries in the dataset


# In[11]:


pbc.isnull().sum() # show the column with no values(null values)


# In[13]:


pbc.describe() # show the count, mean, median, mode and percentile


# In[ ]:





# ● Data Pre-processing ●

# In[14]:


# Our Target Column is the "diagnosis" and remaining columns are the dependent variables.
pbc.shape 


# In[15]:


# Among all the columns "Unnamed: 32" & "id" are not necessary as they are null and invalid columns.
pbc.drop(['Unnamed: 32', 'id'], axis = 1, inplace = True) 


# In[16]:


pbc.shape


# In[17]:


pbc['diagnosis'].unique() # finding the Unique values in the column "diagnosis"


# In[18]:


pbc['diagnosis'].value_counts()
#There are 357-'Benign' and 212-'Malignant'cases in patients


# In[19]:


pbc['diagnosis'].value_counts().plot(kind = 'pie', 
                                    autopct = '%0.2f%%', 
                                    figsize = [5,5], 
                                    explode = [0,0.05], 
                                    colors = ['#00ff99', '#9933ff'], 
                                    shadow = True)

plt.show()


# In[20]:


# As the Target values are in categorical form, lets label encode the vales to numerical data

label_encode = LabelEncoder() # initialising the LabelEncoder

labels = label_encode.fit_transform(pbc['diagnosis']) #Transforming the Categorical to Numerical data

pbc['t_diagnosis'] = labels # Creating a new column


# In[21]:


pbc['t_diagnosis'].value_counts()
# B - 357
# M - 212


# In[22]:


pbc['t_diagnosis'].head()


# In[23]:


# deleting the "diagnosis" from the data set.
pbc.drop(columns='diagnosis', axis=1, inplace=True)


# In[24]:


pbc.columns


# In[25]:


corr = pbc.corr() # Checking for Correlation

kot = corr[corr>=.7]

plt.figure(figsize = [15,10])

sns.heatmap(kot, annot = True, cmap = 'gist_rainbow_r', fmt = '0.2f')

plt.show()


# In[ ]:





# ● Training SVM classifier to predict whether the patient has cancer or not ●

# In[ ]:





# In[26]:


# Train Test Split

X = pbc.drop(columns='t_diagnosis', axis=1)
Y = pbc['t_diagnosis']


# In[27]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify = Y, random_state=3)

print(X.shape, X_train.shape, X_test.shape)


# In[ ]:





# In[28]:


# from sklearn.metrics import accuracy_score 
# from sklearn.svm import SVC 

model = SVC(kernel='linear') # Defining the Model to be Trained
model.fit(X_train, Y_train) 

test_data_prediction = model.predict(X_test) # Predicting

accuracy = accuracy_score(Y_test, test_data_prediction) # Model Accuracy

print('Accuracy score of the ', model, ' = ', accuracy) 


# * Accuracy score of the  SVC(kernel='linear')  =  0.9824561403508771 *

# In[29]:


# B - 357 - 0
# M - 212 - 1
# sample inputs
inp0 = (11.51,23.93,74.52,403.5,0.09261,0.1021,0.1112,0.04105,0.1388,0.0657,0.2388,2.904,1.936,16.97,0.0082,0.02982,0.05738,0.01267,0.01488,0.004738,12.48,37.16,82.28,474.2,0.1298,0.2517,0.363,0.09653,0.2112,0.08732)

inp1 = (7.76,24.54,47.92,181,0.05263,0.04362,0,0,0.1587,0.05884,0.3857,1.428,2.548,19.15,0.007189,0.00466,0,0,0.02676,0.002783,9.456,30.37,59.16,268.6,0.08996,0.06444,0,0,0.2871,0.07039)

inp2 = (20.6,29.33,140.1,1265,0.1178,0.277,0.3514,0.152,0.2397,0.07016,0.726,1.595,5.772,86.22,0.006522,0.06158,0.07117,0.01664,0.02324,0.006185,25.74,39.42,184.6,1821,0.165,0.8681,0.9387,0.265,0.4087,0.124)


# In[30]:


# SVM classifier to predict whether the patient has cancer or not:
# let take inp1 as patient 01 and inp2 as patient 02.


inp1 = (7.76,24.54,47.92,181,0.05263,0.04362,0,0,0.1587,0.05884,0.3857,1.428,2.548,19.15,0.007189,0.00466,0,0,0.02676,0.002783,9.456,30.37,59.16,268.6,0.08996,0.06444,0,0,0.2871,0.07039)
#
inp2 = (20.6,29.33,140.1,1265,0.1178,0.277,0.3514,0.152,0.2397,0.07016,0.726,1.595,5.772,86.22,0.006522,0.06158,0.07117,0.01664,0.02324,0.006185,25.74,39.42,184.6,1821,0.165,0.8681,0.9387,0.265,0.4087,0.124)
# change the input data to a numpy array
input_data_as_numpy_array = np.asarray(inp2)

# reshape the numpy array as we are predicting for one datapoint
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The Breast cancer is Benign')

else:
  print('The Breast Cancer is Malignant')


# In[ ]:


"FRORM THE ABOVE IT IS CLEAR THAT PATIENT 02 HAS Malignant AND PATIENT 01 HAS BENING"


# In[242]:





# ● Assess the correctness in classifying data with respect to efficiency and effectiveness of ●
# ● the SVM classifier in terms of accuracy, precision, sensitivity, specificity and AUC ROC ●

# In[ ]:





# In[31]:


# Accuracy Score for Training Data:
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print(training_data_accuracy)
#
# 0.9604395604395605 - Accuracy Score


# In[33]:


print('Accuracy on Training data : ', round(training_data_accuracy*100, 2), '%')


# In[34]:


# Accuracy on test data

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print(test_data_accuracy)


# In[35]:


print('Accuracy on Test data : ', round(test_data_accuracy*100, 2), '%')


# In[36]:


# Confuaion Matrix:

from sklearn.metrics import confusion_matrix

cf_matrix = confusion_matrix(Y_test, X_test_prediction)

print(cf_matrix)


# In[37]:


# to reveal the TP,TN,FN,FP

tn, fp, fn, tp = cf_matrix.ravel() 

print(tn, fp, fn, tp)


# In[ ]:





# In[44]:


# Plotting Confuaion Matrix on Heat map for better understanding:

plt.figure(figsize = [4,2])
sns.heatmap(cf_matrix, annot=True)
plt.show()


# In[47]:


# Deriving Precision, Recall, F1 Scores:

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

def precision_recall_f1_score(true_labels, pred_labels):

  precision_value = precision_score(true_labels, pred_labels)
  recall_value = recall_score(true_labels, pred_labels)
  f1_score_value = f1_score(true_labels, pred_labels)

  print('Precision =',precision_value)
  print('Recall =',recall_value)
  print('F1 Score =',f1_score_value)
    
#


# In[48]:


# classification metrics for training data

precision_recall_f1_score(Y_train, X_train_prediction)


# In[49]:


# classification metrics for test data

precision_recall_f1_score(Y_test, X_test_prediction)


# In[ ]:


# Comparing the Time Efficiency score of test and train.


# In[64]:


import time

st = time.time()

model1 = SVC(kernel="rbf", degree=1,gamma=0.001, C=100, probability=True)

model1.fit(X_train, Y_train)

print("Train score :", model1.score(X_train, Y_train))
print("Test score :", model1.score(X_test, Y_test))

et = time.time()
print("Time Taken", et-st, "sec")

# -------------------------------------------------------------- #


# In[74]:


# AUC - ROC Curve:

import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score

#model.fit(X_train, Y_train, probability=True)

# Train the model
model.fit(X_train, Y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate the Rank-Based AUC
fpr, tpr, thresholds = roc_curve(Y_test, y_pred)

auc = roc_auc_score(Y_test, y_pred)

# Plot the ROC curve

plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')

plt.show()




# In[ ]:





# In[ ]:


● Tuning the hyperparameters of SVM Classifier provided by the scikit-learn library ●


# In[ ]:





# In[ ]:


# X = pbc.drop(columns='t_diagnosis', axis=1)
# Y = pbc['t_diagnosis']


# In[260]:


X = np.asarray(X)
    
Y = np.asarray(Y)


# In[261]:


# loading the SVC model
model = SVC()


# In[ ]:


# hyperparameters

parameters = {
              'kernel':['linear','poly','rbf','sigmoid'],
              'C':[1, 5, 10, 20]
                }


# In[270]:


classifier = GridSearchCV(model, parameters, cv=5)


# In[ ]:


classifier.fit(X, Y)


# In[ ]:





# In[266]:


classifier.cv_results_


# In[267]:


# Best Parameters:

best_parameters = classifier.best_params_
print(best_parameters)
# {'C': 5, 'kernel': 'linear'}


# In[268]:


# Highest Accuracy:

highest_accuracy = classifier.best_score_
print(highest_accuracy)
# 0.952585002328831


# In[ ]:


# -------------------------------------------------------------------------- #


# In[ ]:





# In[ ]:


Scope:
    
● # Analysing the available data and exploring relationships among given variables
● # Data Pre-processing
● # Training SVM classifier to predict whether the patient has cancer or not
● # Assess the correctness in classifying data with respect to efficiency and effectiveness of
  # the SVM classifier in terms of accuracy, precision, sensitivity, specificity and AUC ROC
● # Tuning the hyperparameters of SVM Classifier provided by the scikit-learn library

