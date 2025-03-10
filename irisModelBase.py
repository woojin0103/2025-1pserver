#2025.3.10
#project2 붓꽃분류기 만들기
# 이용희 교수님과 열심히 만들어보자
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

iris_df = pd.read_csv('iris.csv')
print(iris_df)

Y = iris_df['species']
X = iris_df.drop('species',axis=1)
print(Y)
print(X)

kn = KNeighborsClassifier()
model_kn = kn.fit(X,Y)

rfc = RandomForestClassifier()
model_rfc = rfc.fit(X,Y)

joblib.dump(model_rfc,'model_rfc.pkl')

X_new = np.array([[5,3.4,1.4,0.2]])
prediction = model_kn.predict(X_new)

prediction = model_kn.predict(X_new)
prediction = model_rfc.predict(X_new)
print(prediction)
probability = model_kn.predict_proba(X_new)
probability = model_rfc.predict_proba(X_new)
print(probability)

