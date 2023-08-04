import json
import numpy
import joblib
from azureml.core.model import Model

def init():
	global objModel
	strModelPath = "../models/ReducedLogisticRegression.pkl"        
	objModel = joblib.load(strModelPath)

def run(strRawData, dvarRequestHeader):
	llintData = json.loads(strRawData)["data"]
	aintData = numpy.array(llintData)
	aintResult = objModel.predict(aintData)
	return {"result": aintResult.tolist()}

init()
strTestRow = '{"data":[[0],[1]]}'
dvarRequestHeader = {} 
dlintPredictions = run(strTestRow, dvarRequestHeader)
print("Test row:    ", strTestRow)
print("Test result: ", dlintPredictions)