import numpy as np

x_train=np.array([[-3,7],[1,5],[1,2],[-2,0],[2,3],[-4,0],[-1,1],[1,1],[-2,2],[2,7],[-4,1],[-2,7]])
y_train=np.array([3,3,3,3,4,3,3,4,3,4,4,4])
x_test=np.array([[1,2],[3,9]])

#DECISION TREE
from sklearn import tree
model = tree.DecisionTreeRegressor()
model.fit(x_train,y_train)
model.score(x_train,y_train)
predicted_output = model.predict(x_test)
print(predicted_output)