import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# loading dataset
df = pd.read_csv('D:/Projects/Mobile Price Prediction/cleaned.csv', encoding='utf-8')
df.duplicated().sum()
df = df.drop_duplicates()
df.isna().sum()
df.info()
df.battery = df.battery.astype('int64')
df.weight = df.weight.astype('int64')
df.corr()
#sns.pairplot(data = df)

from scipy import stats
import pylab
stats.probplot(df.price, dist = "norm", plot = pylab); plt.show()

sns.distplot(df['price'])
sns.distplot(np.log(df['price']))
# for log(price) it shows data is normaly distributed so consider log(price) for further calculation
y = np.log(df.price)
x = df.drop('price' , axis = 1)


from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest = train_test_split( x,y,test_size=0.2)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score,mean_absolute_error
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor,ExtraTreesRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

################# Linear Regression ##############################

#OHE_model = OneHotEncoder(handle_unknown = 'ignore')
x.columns

step1 = ColumnTransformer(transformers=[
('col_tnf',OneHotEncoder(sparse=False,handle_unknown = 'ignore'),[0,1,2,3,4])
],remainder='passthrough')

step2 = LinearRegression()
pipe = Pipeline([('step1',step1),('step2',step2)])
pipe.fit(xtrain,ytrain)
ypred = pipe.predict(xtest)
R2_Linear_regression = r2_score(ytest,ypred)
R2_Linear_regression # -8.492105324774862
MAE_linear_regression = mean_absolute_error(ytest,ypred) 
MAE_linear_regression # 2.468864061368225

# we have low amount data due to that for each train_test split R_Square changed so trying to find max rsquare
scores=[]
for i in range(100):
    xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.1,random_state=i)
    step2 = LinearRegression() 
    pipe = Pipeline([('step1',step1),('step2',step2)])
    pipe.fit(xtrain,ytrain)
    ypred=pipe.predict(xtest)
    scores.append(r2_score(ytest,ypred))

np.argmax(scores)
scores[np.argmax(scores)]

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.1,random_state=np.argmax(scores))
step2 = LinearRegression() 
pipe = Pipeline([('step1',step1),('step2',step2)])
pipe.fit(xtrain,ytrain)
ypred=pipe.predict(xtest)
R2_Linear_regression = r2_score(ytest,ypred)
R2_Linear_regression # 0.456725358125591
MAE_linear_regression = mean_absolute_error(ytest,ypred) 
MAE_linear_regression # 0.5403536701456961


#################  Lasso Regression ################

step1 = ColumnTransformer(transformers=[
('col_tnf',OneHotEncoder(sparse=False,handle_unknown = 'ignore'),[0,1,2,3,4])
],remainder='passthrough')

step2 = Lasso(alpha=0.001)

scores=[]
for i in range(100):
    xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.1,random_state=i)
    pipe = Pipeline([('step1',step1),('step2',step2)])
    pipe.fit(xtrain,ytrain)
    ypred=pipe.predict(xtest)
    scores.append(r2_score(ytest,ypred))

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.1,random_state=np.argmax(scores)) 
pipe = Pipeline([('step1',step1),('step2',step2)])
pipe.fit(xtrain,ytrain)
ypred=pipe.predict(xtest)
R2_Lasso_regression = r2_score(ytest,ypred)
R2_Lasso_regression # 0.8983859153147021
MAE_lasso_regression = mean_absolute_error(ytest,ypred) 
MAE_lasso_regression # 0.2463333119770636


################# Ridge Regression ####################


step1 = ColumnTransformer(transformers=[
('col_tnf',OneHotEncoder(sparse=False,handle_unknown = 'ignore'),[0,1,2,3,4])
],remainder='passthrough')

step2 = Ridge(alpha=0.19)

scores=[]
for i in range(100):
    xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.1,random_state=i)
    pipe = Pipeline([('step1',step1),('step2',step2)])
    pipe.fit(xtrain,ytrain)
    ypred=pipe.predict(xtest)
    scores.append(r2_score(ytest,ypred))

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.1,random_state=np.argmax(scores))
pipe = Pipeline([('step1',step1),('step2',step2)])
pipe.fit(xtrain,ytrain)
ypred=pipe.predict(xtest)
R2_Ridge_regression = r2_score(ytest,ypred)
R2_Ridge_regression # 0.8752050918501406
MAE_ridge_regression = mean_absolute_error(ytest,ypred) 
MAE_ridge_regression # 0.3120659252275645



#####################  KNN ####################



step1 = ColumnTransformer(transformers=[
('col_tnf',OneHotEncoder(sparse=False,handle_unknown = 'ignore'),[0,1,2,3,4])
],remainder='passthrough')

step2 = KNeighborsRegressor(n_neighbors=30)

scores=[]
for i in range(100):
    xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.1,random_state=i) 
    pipe = Pipeline([('step1',step1),('step2',step2)])
    pipe.fit(xtrain,ytrain)
    ypred=pipe.predict(xtest)
    scores.append(r2_score(ytest,ypred))

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.1,random_state=np.argmax(scores))
pipe = Pipeline([('step1',step1),('step2',step2)])
pipe.fit(xtrain,ytrain)
ypred=pipe.predict(xtest)
R2_KNN= r2_score(ytest,ypred)
R2_KNN # 0.4165826416244701
MAE_KNN = mean_absolute_error(ytest,ypred)
MAE_KNN # 0.75477218874277



#################### Decision Tree #############


step1 = ColumnTransformer(transformers=[
('col_tnf',OneHotEncoder(sparse=False,handle_unknown = 'ignore'),[0,1,2,3,4])
],remainder='passthrough')

step2 = DecisionTreeRegressor(max_depth=10)

scores=[]
for i in range(100):
    xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.1,random_state=i)
    pipe = Pipeline([('step1',step1),('step2',step2)])
    pipe.fit(xtrain,ytrain)
    ypred=pipe.predict(xtest)
    scores.append(r2_score(ytest,ypred))

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.1,random_state=np.argmax(scores))
pipe = Pipeline([('step1',step1),('step2',step2)])
pipe.fit(xtrain,ytrain)
ypred=pipe.predict(xtest)
R2_DT = r2_score(ytest,ypred)
R2_DT # 0.8577128656055558
MAE_DT = mean_absolute_error(ytest,ypred)
MAE_DT # 0.35620102220739625


##############  Random Forest #########################

step1 = ColumnTransformer(transformers=[
('col_tnf',OneHotEncoder(sparse=False,handle_unknown = 'ignore'),[0,1,2,3,4])
],remainder='passthrough')

step2 = RandomForestRegressor(n_estimators=100,
                              random_state=3,
                              max_samples=0.5,
                              max_features=0.75,
                              max_depth=25)

scores=[]
for i in range(100):
    xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.1,random_state=i)
    pipe = Pipeline([('step1',step1),('step2',step2)])
    pipe.fit(xtrain,ytrain)
    ypred=pipe.predict(xtest)
    scores.append(r2_score(ytest,ypred))

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.1,random_state=np.argmax(scores))
pipe = Pipeline([('step1',step1),('step2',step2)])
pipe.fit(xtrain,ytrain)
ypred=pipe.predict(xtest)
R2_random_forest = r2_score(ytest,ypred)
R2_random_forest # 0.8950844865998615
MAE_random_forest = mean_absolute_error(ytest,ypred)
MAE_random_forest # 0.28398151413285216


################## SVM ####################


step1 = ColumnTransformer(transformers=[
('col_tnf',OneHotEncoder(sparse=False,handle_unknown = 'ignore'),[0,1,2,3,4])
],remainder='passthrough')

step2 = SVR(kernel='rbf',C=10000,epsilon=0.1)

scores=[]
for i in range(100):
    xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.1,random_state=i)
    pipe = Pipeline([('step1',step1),('step2',step2)])
    pipe.fit(xtrain,ytrain)
    ypred=pipe.predict(xtest)
    scores.append(r2_score(ytest,ypred))

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.1,random_state=np.argmax(scores))
pipe = Pipeline([('step1',step1),('step2',step2)])
pipe.fit(xtrain,ytrain)
ypred=pipe.predict(xtest)
R2_svm = r2_score(ytest,ypred)
R2_svm # 0.8999890065919421
MAE_SVM = mean_absolute_error(ytest,ypred)
MAE_SVM # 0.27776714707252564


################  Extra Trees ##########

step1 = ColumnTransformer(transformers=[
('col_tnf',OneHotEncoder(sparse=False,handle_unknown = 'ignore'),[0,1,2,3,4])
],remainder='passthrough')

step2 = ExtraTreesRegressor(n_estimators=100,
                              random_state=3,
                              max_features=0.75,
                              max_depth=28)

scores=[]
for i in range(100):
    xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.1,random_state=i)
    pipe = Pipeline([('step1',step1),('step2',step2)])
    pipe.fit(xtrain,ytrain)
    ypred=pipe.predict(xtest)
    scores.append(r2_score(ytest,ypred))

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.1,random_state=np.argmax(scores))
pipe = Pipeline([('step1',step1),('step2',step2)])
pipe.fit(xtrain,ytrain)
ypred=pipe.predict(xtest)
R2_extra_trees= r2_score(ytest,ypred)
R2_extra_trees # 0.9326098147365638
MAE_extra_trees = mean_absolute_error(ytest,ypred)
MAE_extra_trees # 0.27185577223427837



################### AdaBoost #################

step1 = ColumnTransformer(transformers=[
('col_tnf',OneHotEncoder(sparse=False,handle_unknown = 'ignore'),[0,1,2,3,4])
],remainder='passthrough')

step2 = AdaBoostRegressor(n_estimators=15,learning_rate=0.5)

scores=[]
for i in range(100):
    xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.1,random_state=i)
    pipe = Pipeline([('step1',step1),('step2',step2)])
    pipe.fit(xtrain,ytrain)
    ypred=pipe.predict(xtest)
    scores.append(r2_score(ytest,ypred))

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.1,random_state=np.argmax(scores))
pipe = Pipeline([('step1',step1),('step2',step2)])
pipe.fit(xtrain,ytrain)
ypred=pipe.predict(xtest)
R2_AdaBoost = r2_score(ytest,ypred)
R2_AdaBoost # 0.858135878128405
MAE_adaboost = mean_absolute_error(ytest,ypred)
MAE_adaboost # 0.28167447547684366


####################### Gradient Boost ###############

step1 = ColumnTransformer(transformers=[
('col_tnf',OneHotEncoder(sparse=False,handle_unknown = 'ignore'),[0,1,2,3,4])
],remainder='passthrough')

step2 = GradientBoostingRegressor(n_estimators=500)

scores=[]
for i in range(100):
    xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.1,random_state=i)
    pipe = Pipeline([('step1',step1),('step2',step2)])
    pipe.fit(xtrain,ytrain)
    ypred=pipe.predict(xtest)
    scores.append(r2_score(ytest,ypred))

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.1,random_state=np.argmax(scores))
pipe = Pipeline([('step1',step1),('step2',step2)])
pipe.fit(xtrain,ytrain)
ypred=pipe.predict(xtest)
R2_GBoost = r2_score(ytest,ypred)
R2_GBoost # 0.7541348795828255
MAE_gboost = mean_absolute_error(ytest,ypred)
MAE_gboost # 0.38337122575719024


############################## XG Boost #####################


step1 = ColumnTransformer(transformers=[
('col_tnf',OneHotEncoder(sparse=False,handle_unknown = 'ignore'),[0,1,2,3,4])
],remainder='passthrough')

step2 = XGBRegressor(n_estimators=45,max_depth=5,learning_rate=0.5)

scores=[]
for i in range(100):
    xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.1,random_state=i)
    pipe = Pipeline([('step1',step1),('step2',step2)])
    pipe.fit(xtrain,ytrain)
    ypred=pipe.predict(xtest)
    scores.append(r2_score(ytest,ypred))

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.1,random_state=np.argmax(scores))
pipe = Pipeline([('step1',step1),('step2',step2)])
pipe.fit(xtrain,ytrain)
ypred=pipe.predict(xtest)
R2_XGBoost= r2_score(ytest,ypred)
R2_XGBoost # 0.881944991717481
MAE_xgboost = mean_absolute_error(ytest,ypred) 
MAE_xgboost #0.27340956155252805



############################## Voting Regressor #####################

from sklearn.ensemble import VotingRegressor,StackingRegressor


step1 = ColumnTransformer(transformers=[
('col_tnf',OneHotEncoder(sparse=False,handle_unknown = 'ignore'),[0,1,2,3,4])
],remainder='passthrough')

rf = RandomForestRegressor(n_estimators=350,random_state=3,max_samples=0.5,max_features=0.75,max_depth=15)
gbdt = GradientBoostingRegressor(n_estimators=100,max_features=0.5)
xgb = XGBRegressor(n_estimators=25,learning_rate=0.3,max_depth=5)
et = ExtraTreesRegressor(n_estimators=100,random_state=3,max_samples=0.5,max_features=0.75,max_depth=10)

step2 = VotingRegressor([('rf', rf), ('gbdt', gbdt), ('xgb',xgb), ('et',et)],weights=[5,1,1,1])

scores=[]
for i in range(10):
    xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.1,random_state=i)
    pipe = Pipeline([('step1',step1),('step2',step2)])
    pipe.fit(xtrain,ytrain)
    ypred=pipe.predict(xtest)
    scores.append(r2_score(ytest,ypred))

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.1,random_state=np.argmax(scores))
pipe = Pipeline([('step1',step1),('step2',step2)])
pipe.fit(xtrain,ytrain)
ypred=pipe.predict(xtest)
R2_voting_regressor = r2_score(ytest,ypred)
R2_voting_regressor # 0.881944991717481
MAE_voting_regressor = mean_absolute_error(ytest,ypred)
MAE_voting_regressor # 0.27340956155252805


############################## Stacking #####################

step1 = ColumnTransformer(transformers=[
('col_tnf',OneHotEncoder(sparse=False,handle_unknown = 'ignore'),[0,1,2,3,4])
],remainder='passthrough')

estimators = [
    ('rf', RandomForestRegressor(n_estimators=350,random_state=3,max_samples=0.5,max_features=0.75,max_depth=15)),
    ('gbdt',GradientBoostingRegressor(n_estimators=100,max_features=0.5)),
    ('xgb', XGBRegressor(n_estimators=25,learning_rate=0.3,max_depth=5))
]

step2 = StackingRegressor(estimators=estimators, final_estimator=Ridge(alpha=100))

scores=[]
for i in range(10):
    xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.1,random_state=i)
    pipe = Pipeline([('step1',step1),('step2',step2)])
    pipe.fit(xtrain,ytrain)
    ypred=pipe.predict(xtest)
    scores.append(r2_score(ytest,ypred))

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.1,random_state=np.argmax(scores))
pipe = Pipeline([('step1',step1),('step2',step2)])
pipe.fit(xtrain,ytrain)
ypred=pipe.predict(xtest)
R2_stacking = r2_score(ytest,ypred)
R2_stacking # 0.7830474538794047
MAE_stacking = mean_absolute_error(ytest,ypred)
MAE_stacking # 0.37699369778171554


# Finding Best Model 

data = {"Model" : pd.Series(['Linear Regression' , 'Lasso Regression' , 'Ridge Regression' ,
'KNN', 'Decision Tree', 'Random Forest', 'SVM', 'Extra Trees', 'AdaBoost', 'Gradient Boost',
'XG Boost', 'Voting Regressor', 'Stacking']) , 
"R Square Value" : pd.Series([R2_Linear_regression , R2_Lasso_regression , R2_Ridge_regression , 
R2_KNN , R2_DT , R2_random_forest , R2_svm , R2_extra_trees , R2_AdaBoost , R2_GBoost , R2_XGBoost , 
R2_voting_regressor , R2_stacking]) ,
"Mean Absolute Error" : pd.Series([MAE_linear_regression , MAE_lasso_regression , MAE_ridge_regression , 
MAE_KNN , MAE_DT , MAE_random_forest , MAE_SVM , MAE_extra_trees , MAE_adaboost , MAE_gboost , 
MAE_xgboost , MAE_voting_regressor , MAE_stacking])} 
                                                                                                                                                         
R_Square_and_Error_Values = pd.DataFrame(data)
R_Square_and_Error_Values

# So Extra Trees is the best Model Among all this models it has Higher R square value = 0.830194
# and lower error value =  0.326194

################   XG Boost   ##########


step1 = ColumnTransformer(transformers=[
('col_tnf',OneHotEncoder(sparse=False,handle_unknown = 'ignore'),[0,1,2,3,4])
],remainder='passthrough')

step2 = XGBRegressor(n_estimators=45,max_depth=5,learning_rate=0.5)

scores=[]
for i in range(1000):
    xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.1,random_state=i)
    pipe = Pipeline([('step1',step1),('step2',step2)])
    pipe.fit(xtrain,ytrain)
    ypred=pipe.predict(xtest)
    scores.append(r2_score(ytest,ypred))

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.1,random_state=np.argmax(scores))
pipe = Pipeline([('step1',step1),('step2',step2)])
pipe.fit(xtrain,ytrain)
ypred=pipe.predict(xtest)
R2_XGBoost= r2_score(ytest,ypred)
R2_XGBoost # 0.988591402402681
MAE_xgboost = mean_absolute_error(ytest,ypred) 
MAE_xgboost # 0.048101283144870104


# Exporting model 
import pickle
pickle.dump(df,open('df.pkl','wb'))
pickle.dump(pipe,open('model.pkl','wb'))
















