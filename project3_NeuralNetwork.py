import numpy as np
import pandas as pd

from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix


"""Sacando y ordenando los datos"""
data = pd.read_excel('credit_card_clients.xls')
data = data.drop(['ID'])

#Separate the predictos from the response
X = data.iloc[:,:23]

X = X.astype(float) 
Y = data['Y']
Y=Y.astype('int')

#print(data['Y'].value_counts())

sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=88)
Y = Y.values
X = X.values

sss.get_n_splits(X, Y)
for train_index, test_index in sss.split(X, Y):
	X_train, X_test = X[train_index], X[test_index]
	Y_train, Y_test = Y[train_index], Y[test_index]


"""test with Sklearn"""
clf = MLPClassifier(random_state=1)
clf.fit(X_train, Y_train)   
Y_pred = clf.predict(X_test)

probas = clf.predict_proba(X_test)

accuracy = (Y_pred == Y_test).mean()
print("accuracy of the classifier using Sckit:",accuracy)

conf_mat = confusion_matrix(Y_test, Y_pred)
fig, ax = plt.subplots(figsize=(3,3))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='YlGn')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
print(classification_report(Y_test, Y_pred))


"""----------Without sklearn--------------------------"""
def load_data(N=300):
    rng = np.random.RandomState(0)
    X = rng.randn(N, 2)
    y = np.array(np.logical_xor(X[:, 0] > 0, X[:, 1] > 0), dtype=int)
    y = np.expand_dims(y, 1)
    y_hot_encoded = []

    for x in y:
        if x == 0:
            y_hot_encoded.append([1,0])
        else:
            y_hot_encoded.append([0, 1])
    return X, np.array(y_hot_encoded)

def sigmoid(z, first_derivative=False):
    if first_derivative:
        return z*(1.0-z)
    return 1.0/(1.0+np.exp(-z))

def tanh(z, first_derivative=True):
    if first_derivative:
        return (1.0-z*z)
    return (1.0-np.exp(-z))/(1.0+np.exp(-z))

def predict(data, weights):
    h1 = sigmoid(np.matmul(data, weights[0]))
    logits = np.matmul(h1, weights[1])
    probs = np.exp(logits)/np.sum(np.exp(logits), axis=1, keepdims=True)
    return np.argmax(probs, axis=1)

N = 1
input_dim = int(X_train.shape[1])
hidden_dim = 10
output_dim = 1
num_epochs = 10000
learning_rate= 1e-3
reg_coeff = 1e-6
losses = []
accuracies=[]

#---------------------------------------------------------------------------------------------------------------
# Initialize weights:
np.random.seed(88)
w1 = 2.0*np.random.random((input_dim, hidden_dim))-1.0      #w0=(2,hidden_dim)
w2 = 2.0*np.random.random((hidden_dim, output_dim))-1.0     #w1=(hidden_dim,2)

#Calibratring variances with 1/sqrt(fan_in)
w1 /= np.sqrt(input_dim)
w2 /= np.sqrt(hidden_dim)


for i in range(num_epochs):
    index = np.arange(X_train.shape[0])[:N]
    #is want to shuffle indices: np.random.shuffle(index)

    #---------------------------------------------------------------------------------------------------------------
    # Forward step:
    h1 = sigmoid(np.matmul(X_train[index], w1))                   #(N, 3)
    logits = sigmoid(np.matmul(h1, w2))                     #(N, 2)
    probs = np.exp(logits)/np.sum(np.exp(logits), axis=1, keepdims=True)
    h2 = logits

    #---------------------------------------------------------------------------------------------------------------
    # Definition of Loss function: mean squared error plus Ridge regularization
    L = np.square(Y_train[index]-h2).sum()/(2*N) + reg_coeff*(np.square(w1).sum()+np.square(w2).sum())/(2*N)

    losses.append([i,L])

    #---------------------------------------------------------------------------------------------------------------
    # Backward step: Error = W_l e_l+1 f'_l
    #       dL/dw2 = dL/dh2 * dh2/dz2 * dz2/dw2
    dL_dh2 = -(Y_train[index] - h2)                               #(N, 2)
    dh2_dz2 = sigmoid(h2, first_derivative=True)            #(N, 2)
    dz2_dw2 = h1                                            #(N, hidden_dim)
    #Gradient for weight2:   (hidden_dim,N)x(N,2)*(N,2)
    dL_dw2 = dz2_dw2.T.dot(dL_dh2*dh2_dz2) + reg_coeff*np.square(w2).sum()

    #dL/dw1 = dL/dh1 * dh1/dz1 * dz1/dw1
    #       dL/dh1 = dL/dz2 * dz2/dh1
    #       dL/dz2 = dL/dh2 * dh2/dz2
    dL_dz2 = dL_dh2 * dh2_dz2                               #(N, 2)
    dz2_dh1 = w2                                            #z2 = h1*w2
    dL_dh1 =  dL_dz2.dot(dz2_dh1.T)                         #(N,2)x(2, hidden_dim)=(N, hidden_dim)
    dh1_dz1 = sigmoid(h1, first_derivative=True)            #(N,hidden_dim)
    dz1_dw1 = X_train[index]                                      #(N,2)
    #Gradient for weight1:  (2,N)x((N,hidden_dim)*(N,hidden_dim))
    dL_dw1 = dz1_dw1.T.dot(dL_dh1*dh1_dz1) + reg_coeff*np.square(w1).sum()

    #weight updates:
    w2 += -learning_rate*dL_dw2
    w1 += -learning_rate*dL_dw1
    

y_pred = predict(X_test, [w1, w2])

accuracy = (y_pred == Y_test).mean()
print("accuracy of the classifier:",accuracy)

conf_mat = confusion_matrix(Y_test, y_pred)
fig, ax = plt.subplots(figsize=(3,3))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='YlGn')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
print(classification_report(Y_test, y_pred))

#Getting the graph for the ROC curve
import sklearn.metrics as skm

fig = plt.figure(figsize=(12, 9))

for (_X, _y), label in zip(
    [
        (X_train, Y_train),
        (X_test, Y_test)
    ],
    ["Train", "Test"]
):
    proba = clf.predict_proba(_X)
    fpr, tpr, _ = skm.roc_curve(_y, proba[:, 1])
    roc_auc = skm.auc(fpr, tpr)

    print ("LogisticRegression AUC ({0}): {1}".format(label, roc_auc))

    plt.plot(fpr, tpr, label="{0} (AUC = {1})".format(label, roc_auc), linewidth=4.0)

plt.plot([0, 1], [0, 1], "--", label="Guessing (AUC = 0.5)", linewidth=4.0)

plt.title(r"The ROC curve for Neural Network", fontsize=18)
plt.xlabel(r"False positive rate", fontsize=18)
plt.ylabel(r"True positive rate", fontsize=18)
plt.axis([-0.01, 1.01, -0.01, 1.01])
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(loc="best", fontsize=18)
plt.show()