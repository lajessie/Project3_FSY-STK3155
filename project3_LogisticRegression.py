"""Sacando y ordenando los datos"""
import numpy as np
import pandas as pd
data = pd.read_excel('credit_card_clients.xls') #read the data
data = data.drop(['ID'])

#Separate the predictos from the response
X = data.iloc[:,:23]
X = X.astype(float) 
Y = data['Y']
Y=Y.astype('int')

print(data['Y'].value_counts())

#taking a look to the distribution
import matplotlib.pyplot as plt
import seaborn as sns
sns.countplot(x='Y', data=data, palette='hls')
plt.show()

#split the data set into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

"""---------Logistic Regression code-------------------"""
class LogisticRegression:
    def __init__(self, lr=0.01, num_iter=100000, fit_intercept=True, verbose=False):
        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
        self.verbose = verbose
    
    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    
    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    def __loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    
    def fit(self, X, y):
        if self.fit_intercept:
            X = self.__add_intercept(X)
        
        # weights initialization
        self.theta = np.zeros(X.shape[1])
        
        #Standard Gradient Descent
        for i in range(self.num_iter):
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.lr * gradient
              
    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)
        return self.__sigmoid(np.dot(X, self.theta))
    
    def predict(self, X):
        return self.predict_prob(X).round()
 

#Now instantiate the class and train the model 
model = LogisticRegression(lr=0.1, num_iter=100000)
model.fit(X_train,Y_train)
#Predict with the test set
preds = model.predict(X_test)
#compute the accuracy
accuracy = (preds == Y_test).mean()
print("acc with my lr:",accuracy)
print(model.theta, '\n\n')

from sklearn.metrics import classification_report
from sklearn import metrics
conf_mat = metrics.confusion_matrix(Y_test, preds)
ig, ax = plt.subplots(figsize=(3,3))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='YlGn')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
print(classification_report(Y_test, preds))


"""-----Logistic Regression using Sklear-----"""
from sklearn.linear_model import LogisticRegression

modelS = LogisticRegression(C=1e20)
modelS.fit(X_train, Y_train)
predS = modelS.predict(X_test)

#compute the accuracy 
accuracy = (predS == Y_test).mean()
print("accuracy using sklearn :",accuracy)
#compute the confusion matrix
conf_mat = metrics.confusion_matrix(Y_test, predS)
ig, ax = plt.subplots(figsize=(3,3))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='YlGn')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

#getting the graph of the ROC curve
import sklearn.metrics as skm

fig = plt.figure(figsize=(12, 9))
for (_X, _y), label in zip(
    [
        (X_train, Y_train),
        (X_test, Y_test)
    ],
    ["Train", "Test"]
):
    proba = modelS.predict_proba(_X)
    fpr, tpr, _ = skm.roc_curve(_y, proba[:, 1])
    roc_auc = skm.auc(fpr, tpr)
    print ("LogisticRegression AUC ({0}): {1}".format(label, roc_auc))
    plt.plot(fpr, tpr, label="{0} (AUC = {1})".format(label, roc_auc), linewidth=4.0)

plt.plot([0, 1], [0, 1], "--", label="Guessing (AUC = 0.5)", linewidth=4.0)
plt.title(r"The ROC curve for LogisticRegression", fontsize=18)
plt.xlabel(r"False positive rate", fontsize=18)
plt.ylabel(r"True positive rate", fontsize=18)
plt.axis([-0.01, 1.01, -0.01, 1.01])
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(loc="best", fontsize=18)
plt.show()


