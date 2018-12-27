"""Sacando y ordenando los datos"""
import numpy as np
import pandas as pd
data = pd.read_excel('credit_card_clients.xls')

data = data.drop(['ID'])

#Separate the predictos from the response
X = data.iloc[:,:23]
X = X.astype(float) 
Y = data['Y']
Y=Y.astype('int')

#print(data['Y'].value_counts())

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# Import the model from sklearn
from sklearn.ensemble import RandomForestClassifier
# Instantiate model with 1000 decision trees
rf = RandomForestClassifier(n_jobs=2, random_state=0)
# Train the model on training data
rf.fit(X_train, Y_train);

# Use the forest's predict method on the test data
predictions = rf.predict(X_test)

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

#compute the accuracy, the confusion matrix and the matrix
print("Accuracy:",metrics.accuracy_score(Y_test, predictions))
conf_mat = confusion_matrix(Y_test, predictions)
fig, ax = plt.subplots(figsize=(3,3))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='YlGn')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
print(classification_report(Y_test, predictions))

#getting the graph for the ROC curve
import sklearn.metrics as skm
fig = plt.figure(figsize=(12, 9))

for (_X, _y), label in zip(
    [
        (X_train, Y_train),
        (X_test, Y_test)
    ],
    ["Train", "Test"]
):
    proba = rf.predict_proba(_X)
    fpr, tpr, _ = skm.roc_curve(_y, proba[:, 1])
    roc_auc = skm.auc(fpr, tpr)
    print ("LogisticRegression AUC ({0}): {1}".format(label, roc_auc))
    plt.plot(fpr, tpr, label="{0} (AUC = {1})".format(label, roc_auc), linewidth=4.0)
plt.plot([0, 1], [0, 1], "--", label="Guessing (AUC = 0.5)", linewidth=4.0)

plt.title(r"The ROC curve for Random Forest", fontsize=18)
plt.xlabel(r"False positive rate", fontsize=18)
plt.ylabel(r"True positive rate", fontsize=18)
plt.axis([-0.01, 1.01, -0.01, 1.01])
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(loc="best", fontsize=18)
plt.show()