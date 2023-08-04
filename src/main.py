import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

def SetDefinitions():
	
	global cfltRandomSeed,cstrSourcePath,cstrSourceFile,cstrModelPath,cintTestSize,dstrLookupQuestionTexts,lstrFeatures
	
	cfltRandomSeed = 42 # any number
	cstrSourcePath = "../data/raw/"
	cstrSourceFile = "ACME-HappinessSurvey2020.csv"
	cstrModelPath  = "../models/"
	cintTestSize   = 10
	
	dstrLookupQuestionTexts ={
	    "X1":"my order was delivered on time",
	    "X2":"contents of my order was as I expected",
	    "X3":"I ordered everything I wanted to order",
	    "X4":"I paid a good price for my order",
	    "X5":"I am satisfied with my courier",
	    "X6":"the app makes ordering easy for me"
	}
	
	lstrFeatures = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6']
	
	return cfltRandomSeed,cstrSourcePath,
	
def GetRawData():	
	dfrRaw = pd.read_csv(f"{cstrSourcePath}/{cstrSourceFile}")
	# feature and target
	X = dfrRaw.drop("Y", axis=1)
	y = dfrRaw["Y"]
	return X,y

def SplitData(X,y):
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=cintTestSize, random_state=cfltRandomSeed)
	# scaling
	objStandardScaler = StandardScaler()
	X_train_scaled = objStandardScaler.fit_transform(X_train)
	X_test_scaled  = objStandardScaler.transform(X_test)
	return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled
	
def Baseline(X_train_scaled, y_train, X_test_scaled, y_test):
	objDummyClassifier = DummyClassifier(strategy="most_frequent")
	objDummyClassifier.fit(X_train_scaled, y_train)
	fltAccuracy = objDummyClassifier.score(X_test_scaled, y_test)
	print("Baseline classifier using 'most frequent' strategy:",fltAccuracy)
	strModelName = "Baseline.pkl"
	joblib.dump(value=objDummyClassifier, filename=f"{cstrModelPath}{strModelName}")
	return objDummyClassifier

def Lasso(X_train, y_train,X_test, y_test):
	pipeline = Pipeline([
	    ('scaler',StandardScaler()),
	    ('lasso', LogisticRegression(penalty='l1', solver='liblinear'))
	])
	dvarHyperparameters = {
	    'lasso__random_state': [cfltRandomSeed],
	    'lasso__C': [0.001,0.01,0.1,1,10,100,1000],  # regularization
	    'lasso__max_iter': [100, 200, 500,1000]
	}
	
	# grid search with cross-validation
	objGridSearch = GridSearchCV(pipeline, dvarHyperparameters, cv=5)
	objGridSearch.fit(X_train, y_train)
	
	# get and evaluate best model
	pipBestModel = objGridSearch.best_estimator_
	fltBestScore = objGridSearch.best_score_
	fltTestScore = pipBestModel.score(X_test, y_test)
	a1fltRegressionParameters = pipBestModel.named_steps['lasso'].coef_
	dfrWeights = {
	    "Feature": dstrLookupQuestionTexts.keys(),
	    "Question": dstrLookupQuestionTexts.values(),
	    "Regression weight": a1fltRegressionParameters.flatten()
	}
	strModelName = "Lasso.pkl"
	joblib.dump(value=pipBestModel, filename=f"{cstrModelPath}{strModelName}")
	print("Lasso classifier, full feature array:",fltTestScore)	
	return pipBestModel,fltTestScore,a1fltRegressionParameters

def ReducedModel(X_train, y_train,X_test, y_test):
	X_train_reduced = X_train.copy()
	X_train_reduced["OnTime"] = X_train_reduced["X1"].apply(lambda x: 1 if x == 5 else 0)
	X_train_reduced = X_train_reduced[["OnTime"]]
	X_test_reduced = X_test.copy()
	X_test_reduced["OnTime"] = X_test_reduced["X1"].apply(lambda x: 1 if x == 5 else 0)
	X_test_reduced = X_test_reduced[["OnTime"]]
	
	#  train logistic regression model
	objLogisticRegression = LogisticRegression()
	objLogisticRegression.fit(X_train_reduced, y_train)
	
	# predict
	y_pred = objLogisticRegression.predict(X_test_reduced)
	
	# Calculate the accuracy
	fltAccuracy = accuracy_score(y_test, y_pred)
	strModelName = "ReducedLogisticRegression.pkl"
	joblib.dump(value=objLogisticRegression, filename=f"{cstrModelPath}{strModelName}")	
	print("Logistic regression classifier, reduced feature array:",fltAccuracy)	
	return objLogisticRegression, fltAccuracy
	
def main():
	SetDefinitions()
	X,y = GetRawData()
	X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = SplitData(X,y)
	objDummyClassifier = Baseline(X_train_scaled, y_train, X_test_scaled, y_test)
	objLasso,fltTestScore,a1fltRegressionParameters = Lasso(X_train, y_train,X_test, y_test)
	objLogisticRegression, fltAccuracy = ReducedModel(X_train, y_train,X_test, y_test)

if __name__ == '__main__':
    main()