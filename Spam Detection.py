#!/usr/bin/env python
# coding: utf-8

# * In this project I tried and find out the best fit Machine learning Algorithm for Spam detection
# * By working on this project Im trying to clear my long term doubt of which model is best fit for spam detection
# * So lets work on this project and find out the best Machine Learning model
# * This is Rahul Ashwanth and this is my Spam Detection ML Project
# 

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


spam = pd.read_csv("C:/Users/Rahul/Downloads/spam.csv")


# In[3]:


spam


# # Dataset overview

# In[4]:


spam.head(2)


# In[5]:


spam.tail(2)


# In[6]:


spam.sample(2)


# In[7]:


spam.describe()


# In[8]:


spam.describe(include='object')


# In[9]:


spam.describe


# In[10]:


spam.isnull()


# In[11]:


spam.isnull().sum()


# In[12]:


spam.duplicated()


# In[13]:


spam.duplicated().sum()


# In[14]:


spam.info()


# In[15]:


spam.nunique


# In[16]:


spam.nunique()


# In[17]:


spam.shape


# In[18]:


spam.columns


# In[19]:


spam.dtypes


# In[20]:


spam.notnull()


# In[21]:


spam.notnull().sum()


# In[22]:


spam[spam.duplicated()]


# In[23]:


spam


# In[24]:


spam['Spam']=spam['Category'].apply(lambda x:1 if x=='spam' else 0)


# In[25]:


spam.head(3)


# In[26]:


X=spam['Message']
Y=spam['Spam']


# In[27]:


from sklearn.model_selection import train_test_split



# In[28]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)


# # Naive Bayes

# In[29]:


from sklearn.feature_extraction.text import CountVectorizer


# In[30]:


from sklearn.naive_bayes import MultinomialNB


# In[31]:


from sklearn.pipeline import Pipeline


# In[32]:


sd=Pipeline([
    ('vectorizer',CountVectorizer()),
    ('NB',MultinomialNB())
])


# In[33]:


sd.fit(X_train,Y_train)


# In[34]:


emails=("Congratulation,You are Selected",
       "Apply Now!!! Free offer!!!",
       "Spam Alert!!!",
       "Join Now 50% discount",
       "Win free cash, Login Now",
       "Shortlisted")


# In[35]:


sd.predict(emails)


# In[36]:


Y_pred=sd.predict(X_test)


# In[37]:


sd.score(X_test,Y_test)


# In[38]:


sdpredict=sd.predict(spam)


# In[39]:


print(sdpredict)


# # NAIVE BAYES
# 
# * Best For: Text classification tasks (like spam detection) where the features (words or word frequencies) are treated as independent.
# * Why It Works Well: Naive Bayes is simple, fast, and effective for text classification problems. The Multinomial Naive Bayes variant is particularly well-suited for discrete data like word counts or frequency counts, which is typical for spam detection tasks where you count how often a particular word appears in an email.
# * Advantages:
# #Works well when the data has high dimensionality (like text data with many unique words).
# Simple and fast to train.
# #Performs well even when the assumptions (independence of features) don't hold perfectly.
# * When to Use: If your data has many features (words) and you need a fast, robust baseline model.

# # Model Evaluation

# ## Accuracy 

# In[40]:


from sklearn.metrics import accuracy_score

accuracy=accuracy_score(Y_test,Y_pred)
print(f"Accuracy:{accuracy:.2f}")


# ## Precision

# In[41]:


from sklearn.metrics import precision_score

precision=precision_score(Y_test,Y_pred)
print(f"Precision:{precision:.2f}")


# ## Recall

# In[42]:


from sklearn.metrics import recall_score

recall=recall_score(Y_test,Y_pred)
print(f"Recall:{recall:.2f}")


# ## F1

# In[43]:


from sklearn.metrics import f1_score

f1=f1_score(Y_test,Y_pred)
print(f"F1:{f1:.4f}")


# ## Confusion Matrix

# In[44]:


from sklearn.metrics import confusion_matrix

c__m=confusion_matrix(Y_test,Y_pred)
print("Confusion_matrix1:")
print(c__m)


# ## ROC CURVE and AUC

# In[45]:


from sklearn.metrics import roc_curve,auc
from sklearn.metrics import roc_auc_score

fpr,tpr,threshold = roc_curve(Y_test,Y_pred)
auc_score=auc(fpr,tpr)
print(f"auc:{auc_score:.2f}")


# ## Log Loss

# In[46]:


from sklearn.metrics import log_loss

loss=log_loss(Y_test,Y_pred)
print(f"log_loss:{loss:.4f}")


# ## Cross-Validation

# In[47]:


from sklearn.model_selection import cross_val_score

cv = cross_val_score(sd,X,Y,cv = 5)
print(f"Cross-Validation:{cv}")
print(f"Average CV score:{cv.mean()}")


# # Logistic Regression

# In[48]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


# In[49]:


sp=Pipeline([                                           # Create a pipeline with CountVectorizer and Logistic Regression
    ("Vectorizer",CountVectorizer()),
    ("logreg",LogisticRegression())
])


# In[50]:


sp.fit(X_train,Y_train)                                 # Train the model (assuming you have X_train and y_train)


# In[51]:


emails=("Congratulation,You are Selected",
       "Apply Now!!! Free offer!!!",
       "Spam Alert!!!",
       "Join Now 50% discount",
       "Win free cash, Login Now",
       "shortlisted")


# In[52]:


sp.predict(emails)


# In[53]:


Y_pred = sp.predict(X_test)


# In[54]:


sp.score(X_test,Y_test)


# In[55]:


sppredict=sp.predict(spam)


# In[56]:


print(sppredict)


# # LOGISTIC REGRESSION
# 
# * Best For: Situations where you need to model the relationship between features and the target, and where feature dependencies are present.
# * Why It Works Well: Logistic Regression is a linear classifier that models the probability of a class (e.g., spam or not spam) based on the input features. It’s widely used in spam detection because it works well with high-dimensional data like text, especially when you use regularization (C parameter).
# * Advantages:
# #Flexible and works well in many settings, including when there is a linear relationship between features.
# #Can model feature dependencies (i.e., the interaction between words, which might be more realistic than Naive Bayes' independence assumption).
# #Regularization helps in preventing overfitting when you have many features.
# * When to Use: If you believe the relationship between the features and the target is linear and when regularization is important to prevent overfitting.

# # Model Evaluation 

# ## Accuracy

# In[57]:


from sklearn.metrics import accuracy_score

accuracy = accuracy_score(Y_test,Y_pred)
print(f"Accuracy:{accuracy:.4f}")


# ## Precision

# In[58]:


from sklearn.metrics import precision_score

precision = precision_score(Y_test,Y_pred)
print(f"Precision:{precision:.4f}")


# ## Recall

# In[59]:


from sklearn.metrics import recall_score

recall = recall_score(Y_test,Y_pred)
print(f"Recall:{recall:.4f}")


# ## F1

# In[60]:


from sklearn.metrics import f1_score

f1 = f1_score(Y_test,Y_pred)
print(f"F1:{f1:.4f}")


# ## Confusion Matrix

# In[61]:


from sklearn.metrics import confusion_matrix

cm4 = confusion_matrix(Y_test,Y_pred)
print("Confusion Matrix4")
print(cm4)


# ## Log Loss

# In[63]:


from sklearn.metrics import log_loss

loss = log_loss(Y_test,Y_pred)
print(f"Log_loss:{loss:.4f}")


# ## ROC CURVE AND AUC

# In[64]:


from sklearn.metrics import roc_curve,auc
from sklearn.metrics import roc_auc_score

tpr,fpr,threshold = roc_curve(Y_test,Y_pred)
auc_score = auc(tpr,fpr)
print(f"Auc:{auc_score:.4f}")


# ## Cross-Validation

# In[65]:


from sklearn.model_selection import cross_val_score

cvs = cross_val_score(sp,X,Y,cv = 4)
print(f"Cross-Validation:{cvs}")
print(f"Avg_CV_score:{cvs.mean()}")


# # Random Forest

# In[66]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

rf = Pipeline([
    ('Vectorizer',CountVectorizer()),
    ('rf',RandomForestClassifier())
])


rf.fit(X_train,Y_train)



# In[67]:


emails=("Congratulation,You are Selected",
       "Apply Now!!! Free offer!!!",
       "Spam Alert!!!",
       "Join Now 50% discount",
       "Win free cash, Login Now",
       "shortlisted")

rf.predict(emails)


# In[68]:


Y_pred = rf.predict(X_test)


# In[69]:


rf.score(X_test,Y_test)


# In[70]:


rfpredict = rf.predict(spam)

print(rfpredict)


# # RANDOM FOREST
# 
# * Best For: When you want a non-linear classifier and can handle complex relationships in the data.
# * Why It Works Well: Random Forests are an ensemble of decision trees, and they work by combining multiple weak learners (trees) to make a strong learner. It can handle large feature spaces and is often used when you have structured data with complex relationships.
# * Advantages:
# #Handles both categorical and continuous data well.
# #Does not require feature scaling.
# #Can capture non-linear relationships in the data.
# * When to Use: If you are working with structured data (like user attributes or ratings) alongside text data and need a non-linear model.

# # Model Evaluation

# ## Accuracy

# In[71]:


from sklearn.metrics import accuracy_score

accuracy = accuracy_score(Y_test,Y_pred)
print(f"Accuracy:{accuracy:.3f}")


# ## Precision

# In[72]:


from sklearn.metrics import precision_score

precision = precision_score(Y_test,Y_pred)
print(f"Precision:{precision:.3f}")


# ## Recall

# In[73]:


from sklearn.metrics import recall_score

recall = recall_score(Y_test,Y_pred)
print(f"Recall:{recall:.3f}")


# ## F1

# In[74]:


from sklearn.metrics import f1_score

f1 = f1_score(Y_test,Y_pred)
print(f"F1:{f1:.3f}")


# ## Confusion Matrix

# In[75]:


from sklearn.metrics import confusion_matrix

cm3 = confusion_matrix(Y_test,Y_pred)
print("Confusion_matrix2")
print(cm3)


# ## Log Loss

# In[76]:


from sklearn.metrics import log_loss

loss1 = log_loss(Y_test,Y_pred)
print(f"Log_loss:{loss1:.3f}")


# ## Cross-Validation

# In[77]:


from sklearn.model_selection import cross_val_score

cvss = cross_val_score(rf,X,Y,cv = 5)
print(f"Cross-Validation:{cvss}")
print(f"AVG_CV_SCORE:{cvss.mean()}")


# ## ROC and AUC 

# In[79]:


# from sklearn.metrics import roc_curve,auc
from sklearn.metrics import roc_auc_score

tpr,fpr,threshold = roc_curve(Y_test,Y_pred)
auc_curve = auc(tpr,fpr)
print(f"AUC:{auc_score:.3f}")


# # MODEL EVALUATION
# 
# 
# 
# 

# 
# 
# 
# 
# # Accuracy
# Accuracy is the most straightforward metric. It measures the proportion of correct predictions (both spam and non-spam) out of all predictions.
# 
# Formula: Accuracy =
# True Positives
# +
# True Negatives/
# Total Predictions
# 
#  
# Pros: Easy to understand.
# 
# Cons: It might not be the best metric if the dataset is imbalanced (e.g., if spam is rare compared to non-spam emails).

# # Precision
# Precision measures the accuracy of positive predictions (how many of the emails predicted as spam were actually spam).
# 
# Formula:Precision=
# True Positives/
# True Positives
# +
# False Positives
# 
#  
# Pros: Useful when the cost of false positives (non-spam classified as spam) is high.
# 
# Cons: Does not take false negatives into account.

# # Recall (Sensitivity or True Positive Rate)
# Recall measures the ability of the model to correctly identify spam emails (how many of the actual spam emails were detected).
# 
# Formula:Recall =
# True Positives/
# True Positives
# +
# False Negatives
# 
#  
# Pros: Important when false negatives (spam missed by the model) are more problematic.
# 
# Cons: Does not account for false positives.

# # F1-Score
# F1-Score is the harmonic mean of precision and recall. It balances both the precision and recall metrics into a single value, which makes it particularly useful when you need a balance between the two metrics.
# 
# Formula:F1=2× 
# Precision+Recall/
# Precision×Recall
# 
#  
# Pros: Good for imbalanced datasets where neither precision nor recall should dominate.
# 
# Cons: More complex than accuracy, but more informative when you care about both false positives and false negatives.

# # Confusion Matrix
# A confusion matrix shows the counts of actual vs predicted labels for each class. It gives you a more granular view of how the model is performing.
# 
# Structure:
# True Positives (TP)
# False Negatives (FN)
# ​
#   
#         False Positives (FP)  True Negatives (TN)
# ​
#  
# Pros: Provides a detailed breakdown of model performance, including false positives and false negatives.
# 
# Cons: Does not give a single number for performance (but can be aggregated into other metrics like precision, recall, etc.).

# # ROC Curve and AUC (Area Under the Curve)
# The ROC curve plots the true positive rate (recall) against the false positive rate. The AUC measures the area under the ROC curve, with values closer to 1 indicating a better-performing model.
# 
# Pros: Good for evaluating the performance of a classifier at different thresholds.
# 
# Cons: Requires probabilistic classifiers (i.e., models that output probabilities rather than hard predictions).

# # Log Loss (Logarithmic Loss)
# Log Loss evaluates the accuracy of the probabilistic predictions. It is especially useful when you want to penalize wrong predictions more heavily.
# 
# Pros: It’s useful when your model outputs probabilities, not just hard predictions.
# 
# Cons: Can be more complex and harder to interpret directly.

# # Cross-Validation (CV)
# Cross-validation splits your dataset into multiple subsets (folds), trains the model on some folds, and tests it on the remaining fold. It helps assess how well the model generalizes to unseen data.
# 
# Pros: Provides a more robust estimate of model performance across different subsets of data.
# 
# Cons: More computationally expensive than a simple train-test split.

# # Conclusion: Best Fit for Spam Detection
# * Naive Bayes (MultinomialNB) is often the best starting point for spam detection due to its simplicity, speed, and effectiveness for text classification tasks, especially when features are discrete (word counts).
# * Logistic Regression works well for text classification and can outperform Naive Bayes in some cases, especially if the features have complex relationships.
# * Support Vector Machines (SVM) and Random Forests can be used for more complex models if needed, but they require more computational power and may not be as interpretable as Naive Bayes.
# ### Recommendation from the project based:
# * Start with Multinomial Naive Bayes for a quick, reliable baseline.
# * Logistic Regression could be a good alternative if you want to capture more nuanced relationships between features and the target.
# * Experiment with SVM or Random Forests if your model is underperforming and you suspect more complex patterns exist in the data.# 

# In[ ]:




