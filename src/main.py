import pandas as pd
import numpy as np
import joblib
import os
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

import time; ms = time.time()*1000.0

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import f1_score

import tensorflow as tf
import tensorflow_hub as hub

def SetDefinitions():
	'''
	Sets some global constants.
	
	When       Who What
	2023 08 23 dh  Created
	'''  	
	
	global cblnDebugging  
	global cfltRandomSeed      
	global gcintChannels       
	
	# general
	cfltRandomSeed = 42 # any number
	cblnDebugging = False
	
	# image channels
	gcintChannels = 3
	
	# finalize
	if cblnDebugging:
		print("Debugging comments are on.")
		print("Function SetDefinitions() has run successfully.")
	return
	
def StopWatch(strFormat="",intDigits=0,blnVerbose=True,blnAsFloat=False):
    '''
    No format given: starts the stopwatch.
    Format given: ends the stopwatch, and either prints or returns the elapsed time.
    
    When       Who What
    2022 01 16 dh  Created
    2022 11 07 dh  Format corrected
    '''
    global gfltStopWatchStart

    if strFormat=="":
        gfltStopWatchStart = time.time()
    else:
        
        # calculate difference
        fltStopWatchEnd = time.time()        
        fltSeconds = fltStopWatchEnd - gfltStopWatchStart

        # format
        strFormat = strFormat.lower()
        if strFormat in ["colons","colon","col","c"]:
            strResult = time.strftime('%H:%M:%S', time.gmtime(fltSeconds))
        else:
            if strFormat in ["seconds","second","sec","s"]:
                fltUnits = fltSeconds
                strUnit = "seconds"
            elif strFormat in ["minutes","minute","min","m"]:
                fltUnits = fltSeconds / 60
                strUnit = "minutes"
            elif strFormat in ["hours","hour","hrs","hr","h"]:
                fltUnits = fltSeconds / 3600
                strUnit = "hours"
            else:
                print (f"WARNING: strange parameter {strFormat} in StopWatch. " + 
                       "Formats allowed: colons, seconds, minutes, hours.")  
                return
            strFormat = "{" + f"0:.{intDigits}f" + "}" + " " + strUnit # 2022 11 07 dh adjusted
            strResult = strFormat.format(fltUnits)
        if blnAsFloat:
            return fltUnits
        else:
            if blnVerbose:
                print(strResult)
            else:
                return strResult

def CroppedToSquare(a3intInput, intSquareSide):
    '''
    Crops an image to a square of a given size by keeping the central part.
    The aspect ratio is preserved.
    When       Who What
    2023 08 05 dh  Created
    '''  
    # open image
    pilImage = Image.fromarray(a3intInput)

    # original image size
    intWidth, intHeight = pilImage.size

    # dimensions for cropping and resizing
    intSmallerDimension = min(intWidth, intHeight)
    intLeft = (intWidth - intSmallerDimension) // 2
    intTop = (intHeight - intSmallerDimension) // 2
    intRight = intLeft + intSmallerDimension
    intBottom = intTop + intSmallerDimension

    # crop the image to a square
    pilCroppedImage = pilImage.crop((intLeft, intTop, intRight, intBottom))

    # resize to size required by model
    # - fast: Image.NEAREST
    # - good: Image.ANTIALIAS
    pilResizedImage = pilCroppedImage.resize((intSquareSide, intSquareSide), resample=Image.NEAREST)

    # finalize
    return np.array(pilResizedImage)	

def SequentialModel (strModelName):
    '''
    Loads a transfer learning model.
    Current models available: Inception and MobileNet.
    2022 02 22 dh Created
    '''
    
    if cblnDebugging:
      print("Function SequentialModel() has been entered.")
       
    # source URL elements
    # Python 3 (for Python 2 see ADS-ML course project 4)
    cstrUrlPart1 = "https://tfhub.dev/google/imagenet/"
    cstrUrlPart3 = "/feature_vector/5"
    
    # define model parameters
    strModelNameLower = strModelName.lower()
    if strModelNameLower == "inception":
        intSquareSide = 299
        strUrlPart2 = "inception_v3"
    elif strModelNameLower == "mobilenet":
        intSquareSide = 224
        strUrlPart2 = "mobilenet_v2_100_224"
    else:
        print(f"WARNING: The model {strModelName} does not exist. Possible models: Inception and Mobilenet.")
        return       
    
    # try to load the model
    strModelUrl = cstrUrlPart1 + strUrlPart2 + cstrUrlPart3
    
    try:
        # error level definitions (strings required)
        cstrDisplayAllLogs = "0" 
        cstrDisplayWarningsAndErrorLogs = "1"
        cstrDisplayOnlyErrorLogs = "2"
        cstrDisplayEvenErrors = "3"
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = cstrDisplayOnlyErrorLogs
        
        # get transfer learning model
        objSequentialModel = tf.keras.Sequential([hub.KerasLayer(strModelUrl, trainable=False)])  # can be True, see below.
        
        # reset log level to default
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = cstrDisplayAllLogs
    
    except:
        print("The sequential model cannot be loaded.")
        print(r"Maybe the folder 'C:\Users\domin\AppData\Local\Temp\tfhub_modules' has to be deleted.")
        return
    
    # build model
    lintBatchInputShape = [None, intSquareSide, intSquareSide, gcintChannels]
    objSequentialModel.build(lintBatchInputShape)
    print(f"The sequential model '{strModelName}' has been loaded successfully.")
    return objSequentialModel    

def TrainedSVM(X_train, y_train):
    '''
    Returns a trained SVM, with pre-processing and hyperparameters from last notebook.
    
    When       Who What
    2023 08 04 dh  Created
    2023 08 21 dh  Setting rand_seed
    '''  
    
    # define according to notebook "Binary classifiers"
    objPipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=50,random_state=cfltRandomSeed)),
        ('svm', SVC(probability=True, C=5, gamma=0.01,random_state=cfltRandomSeed))
    ])
    
    # train SVM
    print(f"Training on X_train with shape {X_train.shape}...")  
    StopWatch()
    objPipeline.fit(X_train, y_train)
    
    # finalize
    if cblnDebugging:
      print("Function TrainedSVM() has run successfully.")    
    return objPipeline,StopWatch("seconds",blnAsFloat=True)

def FromDisk(strObjectName, strType="models"):
    '''
    Reads a variable or an object into a "pickle" file.
    Target folder: gcstrPicklePath.
    Filename: the name of the variable or object plus extension "p".
    This function is an extension of VariableFromDisk(strName)    
    
    When       Who What
    2022 01 31 dh  Created
    2023 07 02 dh  Adjusted to Cookiecutter
    2023 08 03 dh  Allowing for direct path access
    '''
    # define path
    if strType == "":
        if "/" in strObjectName:
            strFilename = strObjectName     
        else:
            strFilename = f"{gcstrPicklePath}{strObjectName}.p"
    elif strType.lower() in ["models","model","m"]:
        strSpecialPath = "../models/"
        strFilename = f"{strSpecialPath}{strObjectName}.p"
    elif strType.lower() in ["processed","p"]:
        strSpecialPath = "../data/processed/"
        strFilename = f"{strSpecialPath}{strObjectName}.p"   
    else:
        print(f"Strange type '{strType}' in function FromDisk().")
        
    # open        
    with open(strFilename, 'rb') as objBufferedWriter: # rb = read in binary mode
        return pickle.load(objBufferedWriter)

def TrainSvmOnFourDatasets():
	'''
	Gets the 4 datasets X_train, y_train, X_test, y_test and trains an SVM.
	
	When       Who What
	2023 08 04 dh  Created
	2023 08 23 dh  FromDisk integrated from library
	               Hard-coding exec statement
	'''  
	if cblnDebugging:
		print("Function TrainSvmOnFourDatasets() has been entered.")
		
	# get datasets from disk
	# - exec seems not to work: exec(f"{strDataset} = FromDisk(strPicklePath,strType='')")
	strScriptPath = os.path.abspath(__file__)
	strScriptDirectory = os.path.dirname(strScriptPath)		
	strRoot = f"{strScriptDirectory}/../data/processed/MobileNet/Crop/"
	X_train = FromDisk(f'{strRoot}X_train.p',strType='')
	y_train = FromDisk(f'{strRoot}y_train.p',strType='')
	X_test  = FromDisk(f'{strRoot}X_test.p', strType='')
	y_test  = FromDisk(f'{strRoot}y_test.p', strType='')
	
	# train SVM
	objSVM,fltSeconds = TrainedSVM(X_train, y_train)
	
	# save model
	strModelName = "SVM_MobileNet_Crop.pkl"
	strRoot = f"{strScriptDirectory}/../models/"
	strFilePath = f"{strRoot}{strModelName}"
	joblib.dump(value=objSVM, filename=strFilePath)
	fltFileSizeInMB = os.path.getsize(strFilePath) / 1024 / 1024
	
	# F1
	y_pred = objSVM.predict(X_test)
	fltF1 = f1_score(y_test, y_pred)
	
	# report results
	print("The model has been re-built:")
	print(f"- F1 score on test data:    {round(100*fltF1,1)}%")
	print(f"- time used:                {round(fltSeconds,2)} s")
	print(f"- model size (pickle file): {round(fltFileSizeInMB,2)} MB")
	
	if cblnDebugging:
		print("Function TrainSvmOnFourDatasets() has run successfully.")	
	
def main():
	'''
	Trains an SVM for Apziva project 4: classifying video frames showing book pages being flipped or at rest.
	
	When       Who What
	2023 08 23 dh  Created
	'''
	
	SetDefinitions()
	TrainSvmOnFourDatasets()

	if cblnDebugging:
		print("Function main() has run successfully.")
	
if __name__ == '__main__':
    main()