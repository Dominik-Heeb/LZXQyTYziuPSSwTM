import json
import numpy as np
import joblib
import os
from azureml.core.model import Model

def Init():
	global gobjModel,gstrRootDirectory
	strPickleName = "SVM_MobileNet_Crop.pkl"
	strScriptPath = os.path.abspath(__file__)
	strScriptDirectory = os.path.dirname(strScriptPath)
	gstrRootDirectory = f"{strScriptDirectory}/../"
	strModelPath = f"{gstrRootDirectory}models/{strPickleName}"       
	gobjModel = joblib.load(strModelPath)
	print(f"Model '{strPickleName}' loaded successfully.")
	
def LoadTestData():
	global X_test, y_test
	strDirectory = f"{gstrRootDirectory}data/processed/MobileNet/Crop/"
	strDirectoryAbsolute = os.path.abspath(strDirectory)
	X_test = joblib.load(f"{strDirectory}X_test.p")
	y_test = joblib.load(f"{strDirectory}y_test.p")
	print(f"Test data loaded successfully:")
	print(f"- directory:       {strDirectoryAbsolute}.")
	print(f"- Shape of X_test: {X_test.shape}")
	print(f"- Shape of y_test: {y_test.shape}")

def ArrayToString(a1varSource):
	try:
		a1varSource = a1varSource.astype(int)
	except:
		a1varSource = a1varSource
	return ''.join(map(str, a1varSource))	
	
def Run():
	cintExamples = 50
	if False:
		llintData = json.loads(strRawData)["data"]
		aintData = numpy.array(llintData)
	y_pred = gobjModel.predict(X_test)
	a1blnComparisons = y_pred == y_test
	y_diff = np.where(a1blnComparisons, '-', 'X')	
	intDeviations = len(a1blnComparisons) - np.sum(a1blnComparisons)
	fltDeviations = intDeviations / len(a1blnComparisons)
	fltPercentage = round(100 * fltDeviations,1)
	print("\nPredicted labels\n",ArrayToString(y_pred), sep="")
	print("\nTrue labels\n",     ArrayToString(y_test), sep="")
	print("\nDifferences\n",     ArrayToString(y_diff), sep="")
	print()
	print(f"Deviations: {intDeviations}")
	print(f"Percentage: {fltPercentage}%")

Init()
LoadTestData()
Run()