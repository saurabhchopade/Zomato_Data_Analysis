# Model 4: K Nearest Neighbor Classification

"""
from sklearn import neighbors
knn=neighbors.KNeighborsClassifier()
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
print('K Nearest Neighbor Accuracy:', accuracy_score(y_test, knn_pred)*100)
"""
"""
forest_clf = RandomForestClassifier()
forest_clf.fit(X_train, y_train)
tree.export_graphviz(forest_clf.estimators_[0], out_file='tree_from_forest.dot')
(graph,) = pydot.graph_from_dot_file('tree_from_forest.dot')
graph.write_png(r'D:\tree_from_forest.png')

"""

#tree

"""
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
export_graphviz(ada, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())
"""
"""
plt.plot(ada_pred)
plt.title("Adaboost")
plt.xlabel("amount")
plt.ylabel("Rating")
plt.show()
"""