#IMPORT LIBRARIES
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
#DATASETS
x_train=np.array([[-3,7],[1,5],[1,2],[-2,0],[2,3],[-4,0],[-1,1],[1,1],[-2,2],[2,7],[-4,1],[-2,7]])
y_train=np.array([3,3,3,3,4,3,3,4,3,4,4,4])
x_test=np.array([[1,2],[3,9]])
#MODELS
#model = KNeighborsClassifier()
model = KNeighborsClassifier(n_neighbors=6) #to check upto six neighbours
#MODEL_FIT
model.fit(x_train,y_train)
#MODEL_PREDICT
output = model.predict(x_test)
#OUTPUT
print(output)