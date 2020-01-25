## Machine Learning of the Cleaned Zomato Data
import matplotlib
import pydot as pydot
from jedi.refactoring import inline
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import tree
from sklearn.metrics import f1_score,precision_score,recall_score, confusion_matrix, classification_report,  accuracy_score




import pandas as pd
from sklearn.utils.validation import check_is_fitted

df = pd.read_csv('data/ZomatoindiaCleaned.csv', encoding = 'latin-1', sep = ',')
df.head()


# In[185]:
from sklearn.utils import shuffle
df = shuffle(df) #Shuffling the data
df.head()


### Let us build a Classification Model for Prediciting the rating of a restaurant.
#### Target Value: Rating Text (Values: 1,2,3,4,5)



#Let us Visualize how each feature is correlated to the target value

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
cols = ['Average Cost for two', 'Has Table booking', 'Has Online delivery',  'Is delivering now', 'Price range', 'Votes', 'Rating text',]
cor_matrix = np.corrcoef(df[cols].values.T) # We transpose to get the data by columns. Columns become rows.
sns.set(font_scale=1)
cor_heat_map = sns.heatmap(cor_matrix,
 cbar=True,
 annot=True,
 square=True,
 fmt='.2f',
 annot_kws={'size':10},
 yticklabels=cols,
 xticklabels=cols)
plt.show()


# From the above Heat map, we can see that the Target Value (Rating text) is correlated well with the 'number of votes', 'price Range' and 'average cost for two', with correlation values being 0.42, 0.37 and 0.32 respectively.
#
# We can also draw many other inferences like, Average cost for two and price range has a very high correlation value of 0.83 which says that Average cost for two and price range are highly dependent on each other.

x = df.iloc[:,6:86] #Features
y = df.iloc[:,86] #Target Value - Rating Text
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( x, y, test_size = 0.4, random_state = 1) # Splitting the Data into Training and Test with test size as 20%


# ### Let us Start building Machine Learning Models and predict the accuracy score
def dtada_fun():
    # Model 1: Decision Tree Classification
    global ada_precision, ada_recall, ada_f1score, dtadb_accuracy
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix
    dtree_gini = DecisionTreeClassifier(criterion="gini", random_state=1, max_depth=5, min_samples_leaf=10)
    dtree_gini.fit(X_train, y_train)
    dtree_gini_pred = dtree_gini.predict(X_test)
    print('Decision Tree Accuracy:', accuracy_score(y_test, dtree_gini_pred) * 100)
    cm = confusion_matrix(y_test, dtree_gini_pred)
    from sklearn.metrics import classification_report
    print(classification_report(y_true=y_test,
                                y_pred=dtree_gini_pred))  # Printing the Classification report to view precision, recall and f1 scores

    # Model 2: Decision Tree With Ada Boost

    from sklearn.ensemble import AdaBoostClassifier
    ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=20),
                             n_estimators=60,
                             learning_rate=1.5, algorithm="SAMME")
    ada.fit(X_train, y_train)
    ada_pred = ada.predict(X_test)
    dtadb_accuracy = accuracy_score(y_test, ada_pred) * 100
    print('Decision Tree with Ada Boost Accuracy:', dtadb_accuracy)

    ##precision recalland f1score

    ada_f1score = f1_score(y_test, ada_pred, average="macro")
    ada_recall = recall_score(y_test, ada_pred, average="macro")
    ada_precision = precision_score(y_test, ada_pred, average="macro")

    #################GRAPH
    p = metrics.classification_report(y_test, ada_pred, output_dict=True)

    p1 = pd.DataFrame(p).transpose()
    p1 = p1.drop('support', 1)
    print(p1)
    p1.plot.bar()
    plt.title("DTadaBoost")
    plt.show()

    # Model 2: Random Forest Classification
######################################################################################################################################################
def rf_fun():
    global rf_precision, rf_recall, rf_f1score,rf_accuracy
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=200, oob_score=True)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred) * 100
    print('Random Forest Accuracy:', rf_accuracy)
    # print('Random Forest Out-of-bag score estimate:', rf.oob_score_*100)

    # precision recall f1 score

    rf_f1score = f1_score(y_test, rf_pred, average="macro")
    rf_recall = recall_score(y_test, rf_pred, average="macro")
    rf_precision = precision_score(y_test, rf_pred, average="macro")

    ####################### RF GRAPH
    p = metrics.classification_report(y_test, rf_pred, output_dict=True)
    p1 = pd.DataFrame(p).transpose()
    p1 = p1.drop('support', 1)
    print(p1)
    p1.plot.bar()
    plt.title("Random Forest")
    plt.show()



###########################################################################################################################################################
####################SDD ALGORITHM
def sgd_fun():
    global sgd_precision, sgd_recall, sgd_f1score, sgd_accuracy
    from sklearn.linear_model import SGDClassifier
    sgd = SGDClassifier(loss='modified_huber', shuffle=True, random_state=101)
    sgd.fit(X_train, y_train)
    sgd_pred = sgd.predict(X_test)
    sgd_accuracy = accuracy_score(y_test, sgd_pred) * 100
    print('Stochastic Gradient Descent:', sgd_accuracy)
    #precision recall f1 score
    sgd_f1score = f1_score(y_test, sgd_pred, average="macro")
    sgd_recall = recall_score(y_test, sgd_pred, average="macro")
    sgd_precision = precision_score(y_test, sgd_pred, average="macro")
    ###################SGD graph
    p = metrics.classification_report(y_test, sgd_pred, output_dict=True)

    p1 = pd.DataFrame(p).transpose()
    p1 = p1.drop('support', 1)
    print(p1)
    p1.plot.bar()
    plt.title("stochastic gradient descent")
    plt.show()

#####################################################################Accuracy Single Bar plot
def avgall_fun():
    import numpy as pd, random as pd, pandas as pd
    #data = [dtadb_accuracy, rf_accuracy, sgd_accuracy]
    data=[dtadb_accuracy,rf_accuracy,sgd_accuracy]
    register = 1, 2, 3
    algo = "DTadaboost", "random forest", "SGD",
    plt.figure(figsize=(8, 4))
    b = plt.bar(register, data, width=0.8)
    plt.title("Accuracy of all algorithms", fontsize=20)
    plt.xticks(register, algo)
    plt.legend(b, algo, fontsize=10)
    plt.show()

##############Grouped bar Plot of precision recall and f1score
def compall_fun():
    three = [[ada_precision, ada_recall, ada_f1score], [rf_precision, rf_recall, rf_f1score],[sgd_precision, sgd_recall, sgd_f1score]]

    algorithams = "Adaboost", "Random Forest", "SGD"
    x = np.arange(3)
    plt.bar(x + 0.00, three[0], width=0.25, label="Precision")
    plt.bar(x + 0.25, three[1], width=0.25, label="recall")
    plt.bar(x + 0.5, three[2], width=0.25, label="f1score")
    plt.title("Comparison of all algo precision recall and F1score",fontsize=10)
    plt.xticks(x, algorithams)
    plt.legend(fontsize=15)
    plt.show()

# Model 5: Artificial Neural Networks (Multi Layer Perceptron)
"""
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=500)
mlp.fit(X_train,y_train)
MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=(13, 13, 13), learning_rate='constant',
       learning_rate_init=0.001, max_iter=300, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=None,
       shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
       verbose=False, warm_start=False)

ann_pred = mlp.predict(X_test)
ann_accuracy = accuracy_score(y_test, ann_pred)
print('Artificial Neural Network Accuracy:', ann_accuracy*100)

"""




# # Conclusions:
# 1. Data Analysis Showed that Rating is dependent on Average cost of two and also number of votes
# 2. Correlation Matrix showed that the Average cost for two and price range are highly dependent on each other
# 3. Correlation Matrix also showed that Rating text is correlated well with the 'number of votes', 'price Range' and 'average cost for two'
# 4. Of all the 5 Models, the best accuracy that we are getting is for the decision tree classifier with the accuracy of  around 67%. This means that the Decision Tree model is predicting nearly 67% of the test data accurately
# 5. Future works include: Hyper Parameter Tuning and using other boosting techniques like Xgboost
# 6. Can improve the accuracy by converting the data into a binary classification problem and combining it


