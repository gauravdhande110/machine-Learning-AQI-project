# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv('fillter_data_aqi.csv',usecols = ['CO','no2','pm2_5','AQI'])


X = df.iloc[:, :-1].values
y = df.iloc[:,3:4].values



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Decision tree

from sklearn.tree import DecisionTreeRegressor
regressordc = DecisionTreeRegressor(random_state = 0)
regressordc.fit(X_train, y_train)

# Predicting a new result
y_pred_dc = regressordc.predict(X_test)

#Multi linear regression

from sklearn.linear_model import LinearRegression
regressorml = LinearRegression()
regressorml.fit(X_train, y_train)
y_pred_ml = regressorml.predict(X_test)

# polynomial regression

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(X_train)
poly_reg.fit(X_poly, y_train)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y_train)
y_pred_ply = lin_reg_2.predict(poly_reg.fit_transform(X_test))


# SVRegressor
from sklearn.svm import SVR

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X_train)
Xt = sc_X.fit_transform(X_test)

regressorsvr = SVR(kernel = 'rbf')
regressorsvr.fit(X, y_train)
y_pred_svr = regressorsvr.predict(Xt) 
# randomforest regressor  

from sklearn.ensemble import RandomForestRegressor
regressorrf = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressorrf.fit(X_train, y_train)
y_pred_rf = regressorrf.predict(X_test)



# Visualising the Decision Tree Regression results (higher resolution)
plt.title('  Actual Data')
plt.xlabel('x1(yelllow)=CO x2(Red)=NO2 x3(blue) = PM2.5 ) ', fontsize=10)
plt.ylabel('AQI', fontsize=10)
plt.scatter(X_test[:,0:1],y_test,s = 1,c ='y') 
plt.scatter(X_test[:,1:2],y_test,s = 1,c ='r') 
plt.scatter(X_test[:,2:3],y_test,s = 1,c ='b') 
#plt.scatter(X_test[:,0:1],y_test,s = 1,c ='y')
plt.xlim(0,600)
plt.ylim(0,1000)
#plt.legend('SO2','NO2','PM2.5')
plt.savefig('act.png') 
plt.show()
plt.clf()
plt.title('Decision Tree regressor predicted data')
plt.xlabel('x1(yelllow)=CO x2(Red)=NO2 x3(blue) = PM2.5 ) ', fontsize=10)
plt.ylabel('AQI', fontsize=10)
plt.scatter(X_test[:,0:1],y_pred_dc,s = 1,c ='y') 
plt.scatter(X_test[:,1:2],y_pred_dc,s = 1,c ='r') 
plt.scatter(X_test[:,2:3],y_pred_dc,s = 1,c ='b') 
plt.xlim(0,600)
plt.ylim(0,1000) 
#plt.legend('SO2','NO2','PM2.5')
plt.savefig('DecisionTreepredicted.png') 
plt.show()

plt.clf()
plt.title('multi linear regressor  predicted data')
plt.xlabel('x1(yelllow)=CO x2(Red)=NO2 x3(blue) = PM2.5 ) ', fontsize=10)
plt.ylabel('AQI', fontsize=10)
plt.scatter(X_test[:,0:1],y_pred_ml,s = 1,c ='y') 
plt.scatter(X_test[:,1:2],y_pred_ml,s = 1,c ='r') 
plt.scatter(X_test[:,2:3],y_pred_ml,s = 1,c ='b') 
plt.xlim(0,600)
plt.ylim(0,1000) 
#plt.legend('SO2','NO2','PM2.5')
plt.savefig('multilinearpredicted.png') 
plt.show()

plt.clf()
plt.title('polynomial regressor predicted data')
plt.xlabel('x1(yelllow)=CO x2(Red)=NO2 x3(blue) = PM2.5 ) ', fontsize=10)
plt.ylabel('AQI', fontsize=10)
plt.scatter(X_test[:,0:1],y_pred_ply,s = 1,c ='y') 
plt.scatter(X_test[:,1:2],y_pred_ply,s = 1,c ='r') 
plt.scatter(X_test[:,2:3],y_pred_ply,s = 1,c ='b') 
plt.xlim(0,600)
plt.ylim(0,1000) 
#plt.legend('SO2','NO2','PM2.5')
plt.savefig('polynomialpredicted.png') 
plt.show()



plt.clf()
plt.title('SVR regressor predicted data')
plt.xlabel('x1(yelllow)=CO x2(Red)=NO2 x3(blue) = PM2.5 ) ', fontsize=10)
plt.ylabel('AQI', fontsize=10)
plt.scatter(X_test[:,0:1],y_pred_svr,s = 1,c ='y') 
plt.scatter(X_test[:,1:2],y_pred_svr,s = 1,c ='r') 
plt.scatter(X_test[:,2:3],y_pred_svr,s = 1,c ='b') 
plt.xlim(0,600)
plt.ylim(0,1000) 
#plt.legend('SO2','NO2','PM2.5')
plt.savefig('SVRpredicted.png') 
plt.show()

plt.clf()
plt.title('RandomForest regressor predicted data')
plt.xlabel('x1(yelllow)=CO x2(Red)=NO2 x3(blue) = PM2.5 ) ', fontsize=10)
plt.ylabel('AQI', fontsize=10)
plt.scatter(X_test[:,0:1],y_pred_rf,s = 1,c ='y') 
plt.scatter(X_test[:,1:2],y_pred_rf,s = 1,c ='r') 
plt.scatter(X_test[:,2:3],y_pred_rf,s = 1,c ='b') 
plt.xlim(0,600)
plt.ylim(0,1000) 
#plt.legend('SO2','NO2','PM2.5')
plt.savefig('rfpredicted.png') 
plt.show()


plt.clf()
#from sklearn.metrics import mean_absolute_error
#print(mean_absolute_error(y_test, y_pred))
#from sklearn.metrics import mean_squared_error
#print(mean_squared_error(y_test, y_pred))
from sklearn.metrics import r2_score
print(r2_score(y_test, y_pred_dc))
print(r2_score(y_test, y_pred_ml))
print(r2_score(y_test, y_pred_ply))
print(r2_score(y_test, y_pred_svr))
print(r2_score(y_test, y_pred_rf))




label=['decision tree','multilinear','polynomial','SVR',]

acc= [
r2_score(y_test, y_pred_dc)*100,
r2_score(y_test, y_pred_ml)*100,
r2_score(y_test, y_pred_ply)*100,
r2_score(y_test, y_pred_svr)*100,   
#r2_score(y_test, y_pred_rf)*100
]
#plt.figure(figsize=(15,15))

#print(acc)
index = np.arange(len(label))
plt.ylim(0,110) 
plt.bar(index,acc, color=['green','red','cyan','blue','black'])
plt.xlabel('R2_score', fontsize=10)
plt.ylabel('In Percentage', fontsize=10)
plt.xticks(index, label, fontsize=10, rotation=30)
plt.title('Regression Algorithm COMPARISION R2_SCORE')
plt.savefig('reportregression.png')
plt.show()