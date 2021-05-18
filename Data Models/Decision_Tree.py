# importing the libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import confusion_matrix,accuracy_score, classification_report
from sklearn.metrics import roc_curve, roc_auc_score
from graphviz import Source
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
# importing the dataset
df = pd.read_csv('finalmerge.csv')
# creating the tree
df = df.fillna(0)
df.astype(int)
#variables = list(df.columns[0:2])
variables = list(df.columns[:3])
y = df['Immunity']
X = df[variables]
Tree = tree.DecisionTreeClassifier(max_depth=4)
Tree = Tree.fit(X, y)
#clf = tree.DecisionTreeClassifier()
#clf = clf.fit(iris.data, iris.target)
dot_data_exp = tree.export_graphviz(Tree, out_file = None, feature_names = X.columns, class_names= ['0','1'], filled = True, rounded = True, special_characters = True)
# visualizing the tree
graph = Source(dot_data_exp)
graph.render('diabetes')
graph.view()

# training tree
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

bc_tree = tree.DecisionTreeClassifier(criterion = 'entropy').fit(X_train, y_train)

# calculating the prdiction
bc_pred = bc_tree.predict(X_test)
# evaluting the scores
bc_tree.score(X_test, y_test)
# creating the confusion/error matrix
accuracy = accuracy_score(y_test,bc_pred)
report = classification_report(y_test,bc_pred)
print(accuracy)
print(report)
cm = confusion_matrix(y_test, bc_pred)
print(cm)
# visualizing the error matrix
plt.imshow(cm, cmap = 'binary', interpolation = 'None')
'''fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels(['a'])
ax.set_yticklabels(['b'])
plt.xlabel('Predicted')
plt.ylabel('True')'''
plt.show()

# ## Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(X_train), y_train
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Decision Tree Classification (Training set)')
plt.xlabel('Total Population')
plt.ylabel('Housing Units')
plt.legend()
plt.show()

# ## Visualising the Testing set results
from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(X_test), y_test
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Decision Tree Classification (Test set)')
plt.xlabel('Total Population')
plt.ylabel('Housing Units')
plt.legend()
plt.show()
