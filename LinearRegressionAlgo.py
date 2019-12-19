from sklearn import linear_model
import numpy as np
x_train=np.array([[-3,7],[1,5],[1,2],[-2,0],[2,3],[-4,0],[-1,1],[1,1],[-2,2],[2,7],[-4,1],[-2,7]])
y_train=np.array([3,3,3,3,4,3,3,4,3,4,4,4])
x_test=np.array([[1,2],[3,9]])
linear=linear_model.LinearRegression()
linear.fit(x_train,y_train)
linear.score(x_train,y_train)
print('Coeffient:\n',linear.coef_)
print('Intercept:\n',linear.intercept_)
predicted=linear.predict(x_test)
print(predicted)