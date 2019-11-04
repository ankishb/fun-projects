
import pandas as pd
import numpy as np


sub = pd.read_csv('fun-projects/new/0e7cfe07-39e6-4df4-ab87-eb77cec39a9cConfResult.csv')
test = pd.read_csv('fun-projects/new/db64ebc4-a9b9-42b6-b08b-24eb4ce4bbacFlickBayTest.csv')
train = pd.read_csv('fun-projects/new/2698ce10-d2a3-4e23-9f33-ed5b3716ba9eFlickBayTrain.csv')
print(train.shape, test.shape, sub.shape)


from sklearn.model_selection import train_test_split as tts

X_train, X_test, y_train, y_test = tts(train.drop(['serial','rating'], axis=1), train['rating'], test_size=0.25)
X_train.shape, y_train.shape, X_test.shape, y_test.shape




from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor


from sklearn.metrics import r2_score
dtr = DecisionTreeRegressor(criterion='mae', max_depth=4, max_features=10,  min_samples_leaf=2, min_samples_split=2, random_state=1234, max_leaf_nodes=None)

# criteria: ['mae','mse']
rf_reg = RandomForestRegressor(n_estimators=100, criterion='mae', max_depth=6, max_features=7, min_samples_leaf=2, n_jobs=-1, min_samples_split=2,  bootstrap=True, max_leaf_nodes=None, oob_score=False,verbose=1, random_state=1234)
rf_reg.fit(X_train, y_train)
#rf_reg.score(X_test, y_test)

#dt_ada = DecisionTreeClassifier(max_depth=4, min_samples_leaf=2)
#ada = AdaBoostRegressor(base_estimator=None, n_estimators=50, loss='square', learning_rate=0.05, random_state=1234)
#ada.fit(X_train, y_train)


pred = rf_reg.predict(test.drop(['serial'], axis=1))
final = pd.DataFrame()
final['serial'] = test['serial']
final['rating'] = pred
final.to_csv('final_sub.csv', index=None)