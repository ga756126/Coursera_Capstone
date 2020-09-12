from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
Loan_Classiffier = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
Loan_Classiffier # it shows the default parameters
#Fit the traingin sets
Loan_Classiffier.fit(X_trainset,y_trainset)
#Prediction
predTree = Loan_Classiffier.predict(X_testset)
print (predTree [0:5])
print (y_testset [0:5])

from sklearn import metrics
import matplotlib.pyplot as plt
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))



import scipy.optimize as opt
import pylab as pl
from sklearn import svm
clf = svm.SVC(kernel='rbf', gamma ='scale')
clf.fit(X_train, y_train)
yhat = clf.predict(X_test)
yhat [0:5] 


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
LR = LogisticRegression(C=0.01, solver='liblinear').fit(Test_X,test_y)
LR6`	 



yhat1 = LR.predict(Test_X)
yhat1
yhat_prob = LR.predict_proba(Test_X)
yhat_prob

from sklearn.metrics import jaccard_similarity_score
jaccard_similarity_score(test_y, yhat1)
from sklearn.metrics import log_loss
log_loss(test_y, yhat_prob)