import sys
import cPickle as pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cross_validation import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier



pd.set_option('display.max_columns', None)

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# dict to dataframe
df = pd.DataFrame.from_dict(data_dict, orient='index')
df.replace('NaN', np.nan, inplace = True)

df.info()
df.poi[df.poi==1]
df.plot.scatter(x = 'salary', y = 'bonus')
df['salary'].idxmax()
df.drop('TOTAL', inplace = True)
df.plot.scatter(x = 'salary', y = 'bonus')
df['fraction_from_poi'] = df['from_poi_to_this_person'] / df['to_messages']
df['fraction_to_poi'] = df['from_this_person_to_poi'] / df['from_messages']

ax = df[df['poi'] == False].plot.scatter(x='fraction_from_poi', y='fraction_to_poi', color='blue', label='non-poi')
df[df['poi'] == True].plot.scatter(x='fraction_from_poi', y='fraction_to_poi', color='red', label='poi', ax=ax)


### The first feature must be "poi".
features_list = ['poi', 'salary', 'bonus', 'long_term_incentive', 'deferred_income', 'deferral_payments',
                 'loan_advances', 'other', 'expenses', 'director_fees', 'total_payments', 
                 'exercised_stock_options', 'restricted_stock', 'restricted_stock_deferred', 
                 'total_stock_value', 'to_messages', 'from_messages', 'from_this_person_to_poi', 
                 'from_poi_to_this_person', 'shared_receipt_with_poi', 'fraction_from_poi', 'fraction_to_poi']
data = featureFormat(my_dataset, features_list)

labels, features = targetFeatureSplit(data)

### split data into training and testing datasets
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.5, random_state=1)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from time import time
t0 = time()

clf = DecisionTreeClassifier(random_state=2)
clf.fit(features_train,labels_train)
pred= clf.predict(features_test)
print("Accuracy: ", accuracy_score(labels_test, pred))
print("Precision: ", precision_score(labels_test, pred))
print("Recall: ", recall_score(labels_test, pred))

print("Decision tree algorithm time:", round(time()-t0, 3), "s")



importances = clf.feature_importances_
import numpy as np
indices = np.argsort(importances)[::-1]
print('Feature Ranking: ')
for i in range(16):
    print("{} feature {} ({})".format(i+1,features_list[i+1],importances[indices[i]]))
    
features_list2 = ["salary", "bonus", "fraction_from_poi", "fraction_to_poi", 'deferral_payments', \
                 'total_payments', 'loan_advances', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value']
data2 = featureFormat(my_dataset, features_list2)

 
labels2, features2 = targetFeatureSplit(data2)

### split data into training and testing datasets
from sklearn import cross_validation
features_train2, features_test2, labels_train2, labels_test2 = cross_validation.train_test_split(features2, labels2, test_size=0.5, random_state=1)

t0 = time()

clf = DecisionTreeClassifier(random_state=2)
clf.fit(features_train2,labels_train2)
pred = clf.predict(features_test2)
print("Accuracy: ", accuracy_score(labels_test2, pred))
# print("Precision: ", precision_score(labels_test2, pred))
# print("Recall: ", recall_score(labels_test2, pred))

print("Decision tree algorithm time:", round(time()-t0, 3), "s")

features_list3 = ["poi", "fraction_from_poi", "fraction_to_poi", "shared_receipt_with_poi"]

data3 = featureFormat(my_dataset, features_list3)

labels3, features3 = targetFeatureSplit(data3)

### split data into training and testing datasets
from sklearn import cross_validation
features_train3, features_test3, labels_train3, labels_test3 = cross_validation.train_test_split(features3, labels3, test_size=0.5, random_state=1)

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()

clf.fit(features_train3,labels_train3)
pred= clf.predict(features_test3)
print("Accuracy: ", accuracy_score(labels_test3, pred))
print("Precision: ", precision_score(labels_test3, pred))
print("Recall: ", recall_score(labels_test3, pred))

print("Logistic Regression algorithm time:", round(time()-t0, 3), "s")


t0 = time()

clf = DecisionTreeClassifier(random_state=4)
clf.fit(features_train3,labels_train3)
pred= clf.predict(features_test3)
print("Accuracy: ", accuracy_score(labels_test3, pred))
print("Precision: ", precision_score(labels_test3, pred))
print("Recall: ", recall_score(labels_test3, pred))

print("Decision tree algorithm time:", round(time()-t0, 3), "s")

t0 = time()

clf = GaussianNB()
clf.fit(features_train3, labels_train3)
pred = clf.predict(features_test3)
print("Accuracy: ", accuracy_score(labels_test3, pred))
print("Precision: ", precision_score(labels_test3, pred))
print("Recall: ", recall_score(labels_test3, pred))

print("NB algorithm time:", round(time()-t0, 3), "s")

# 特征列表
features_list = ["poi", "fraction_from_poi", "fraction_to_poi", "shared_receipt_with_poi"]

# 保存经过预处理的数据集
my_dataset = data_dict

# 原数据集为字典类型，python字典不能直接读入到sklearn分类或回归算法中，我编写了一些辅助函数`featureFormat()`，
# 它可以获取特征名的列表和数据字典，然后返回`numpy`数组,如果特征没有某个特定人员的值(即NaN)，此函数还会用 0替换特征值。
data = featureFormat(my_dataset, features_list)

# 分开类标号和特征
labels, features = targetFeatureSplit(data)


# 交叉验证
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.5, random_state=1)

### use KFold for split and validate algorithm
from sklearn.cross_validation import KFold
kf=KFold(len(labels),3)
for train_indices, test_indices in kf:
    # make training and testing sets
    features_train= [features[ii] for ii in train_indices]
    features_test= [features[ii] for ii in test_indices]
    labels_train=[labels[ii] for ii in train_indices]
    labels_test=[labels[ii] for ii in test_indices]

    
# 决策树
from sklearn.tree import DecisionTreeClassifier

t0 = time()

clf = DecisionTreeClassifier(random_state=0)
clf.fit(features_train,labels_train)
score = clf.score(features_test,labels_test)
print('accuracy before tuning ', score)

print("Decision tree algorithm time:", round(time()-t0, 3), "s")

### use manual tuning parameter min_samples_split
t0 = time()
clf2 = DecisionTreeClassifier(min_samples_split=5, random_state=0)
clf2 = clf2.fit(features_train,labels_train)
pred2 = clf2.predict(features_test)
print("done in %0.3fs" % (time() - t0))

print("Validating algorithm:")
print("accuracy after tuning = ", accuracy_score(labels_test, pred2))
print('precision = ', precision_score(labels_test,pred2))
print('recall = ', recall_score(labels_test,pred2))
#############################################################################



### dump your classifier, dataset and features_list so
### anyone can run/check your results
pickle.dump(clf, open("my_classifier.pkl", "wb") )
pickle.dump(data_dict, open("my_dataset.pkl", "wb") )
pickle.dump(features_list, open("my_feature_list.pkl", "wb"))