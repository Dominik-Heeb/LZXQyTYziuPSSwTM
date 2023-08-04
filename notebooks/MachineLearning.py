import Utilities as u

import matplotlib.pyplot as plt
import numpy as np
def DrawDistributionPlots(dfrDataSource,lstrColumns,
	                          intDiagramColumns = 5, intBins=-1, fltTotalWidth = 12, fltTotalHeight = -1,
	                          llfltVerticalLines = []):
    '''
    Draws a table of distribution plots, for numeric columns of a dataframe.
    Allows for vertical lines, e.g. to mark boundaries of z-scores.
    2022 02 02 dh Created
    2022 02 10 dh Allow for a single diagram
    2022 02 20 dh Vertical lines added
    '''
    if intBins == -1:
        intBins = round(np.sqrt(dfrDataSource.shape[0]))

    intSourceColumns = len(lstrColumns)
    intDiagramRows = (intSourceColumns - 1) // intDiagramColumns + 1
    if fltTotalHeight == -1:
        fltTotalHeight = intDiagramRows * 1.7
    
    # no empty columns
    if intDiagramColumns > len(lstrColumns):
        intDiagramColumns = len(lstrColumns)
  
    if len(lstrColumns) == 1: # single plot
        strColumn = lstrColumns[0]
        plt.figure(figsize=(fltTotalWidth, fltTotalHeight))
        plt.hist(dfrDataSource[strColumn].dropna(), bins=intBins, color="orange", edgecolor="black")
        plt.title(strColumn) 
        try:
            lfltVerticalLines = llfltVerticalLines[0]
        except:
            lfltVerticalLines = []   
        fltYMax = plt.ylim()[1]
        for fltX in lfltVerticalLines:
            plt.vlines(fltX,0,fltYMax, colors=["darkblue"])
    else:                     # several plots
        intPlotPointer = 0
        fig, axes = plt.subplots(nrows=intDiagramRows, ncols=intDiagramColumns, figsize=(fltTotalWidth, fltTotalHeight))
        for strColumn, objAxesSubplot in zip(lstrColumns, axes.ravel()): # ravel() flattens an area
            try:
                lfltVerticalLines = llfltVerticalLines[intPlotPointer]
            except:
                lfltVerticalLines = []
            objAxesSubplot.hist(dfrDataSource[strColumn].dropna(), bins=intBins, color="orange", edgecolor="black")
            objAxesSubplot.set_title(strColumn)
            
            fltYMax = objAxesSubplot.get_ylim()[1]
            for fltX in lfltVerticalLines:
                objAxesSubplot.vlines(fltX,0,fltYMax, colors=["darkblue"])
            intPlotPointer += 1
    
    plt.tight_layout() # to avoid overlapping with the labels
    plt.show()
    
import numpy as np
from sklearn.linear_model import LinearRegression
def LinearRegressionOnSeries (dfrSource,strXColumn, strYColumn):
    '''
    Calculates linear regression on two columns.
    Returns r squared plus starting point and end point of the regression line.
    2022 02 21 dh Created.
    '''    
    afltX = dfrSource[strXColumn].to_numpy().reshape(-1, 1)
    afltY = dfrSource[strYColumn].to_numpy()
    objLinearRegressionModel = LinearRegression().fit(afltX,afltY)
    fltRSquared = objLinearRegressionModel.score(afltX,afltY)
    fltIntercept = objLinearRegressionModel.intercept_
    fltSlope = objLinearRegressionModel.coef_[0]
    fltXMin = np.min(afltX)
    fltXMax = np.max(afltX)
    afltXExtrema = np.array([fltXMin,fltXMax])
    fltYLeft, fltYRight = objLinearRegressionModel.predict(afltXExtrema.reshape(-1, 1))
    return fltRSquared,[fltXMin,fltXMax],[fltYLeft, fltYRight]    
    
from scipy import stats  
import numpy as np
def IsNormallyDistributed (varData, fltThresholdForP = 0.01):
    '''
    Returns p value of a test on normal distribution.
    Method: D'Agostino, R. B. (1971), "An omnibus test of normality for moderate and large sample size". Biometrika, 58, 341-348
    2022 03 05 dh Created
    '''
    fltK2,fltPValue= stats.normaltest(varData) # K2: k statistic, not used here
    return fltPValue > fltThresholdForP 
if False:
    lfltTester = list(np.random.normal(loc=10, scale=2, size=20)) + [0]
    print(IsNormallyDistributed(lfltTester)) 
def HighLevelFeaturesOnly(dfrSource):
    '''
    Extracts high-level image features (HLF) from a dataframe.
    The HLF's must be tagged by "hlf_".
    2022 03 09 dh Created
    '''
    cstrTag = "hlf_"
    dfrHlfOnly = dfrSource.copy()
    for strColumn in dfrHlfOnly.columns:
        if not strColumn.startswith(cstrTag):
            dfrHlfOnly.drop(strColumn,axis=1, inplace=True)
    return dfrHlfOnly   
    
from sklearn.metrics import mean_squared_error as MSE
def RMSE(y_true,y_pred):
    '''
    Calculates root of mean squared error.
    2022 03 10 dh Created
    '''
    return MSE(y_true,y_pred,squared=False)
if False:
    y_true = [1,2,3,4,5]
    y_pred = [1,2,3,4,15]
    print("MSE", MSE (y_true,y_pred))
    print("RMSE",RMSE(y_true,y_pred)) 
    
def HlfTargetPropulsion(strTransferLearningModel,strTarget,blnVerbose=False):
    '''
    Retrieves data from disk, reduces data to high-level features (HLF's) and assigns it to X and y.
    Additionally returns the propulsion type.
    2022 03 10 dh Created
    '''
    # translation table
    dintCodes = {'steam': 0, 'electric': 1, 'diesel': 2}
    lstrDatasets = ["train","valid","test"]
    
    # get data
    dfrScaled = u.FromDisk(f"dfrScaled{strTransferLearningModel}")
    
    # HLF's only
    X,y,daintPropulsion = {},{},{}
    
    if blnVerbose:
        print("Shapes of datasets:")    
    for strDataset in lstrDatasets:
        dfrSingleDataset = dfrScaled[dfrScaled["dataset"] == strDataset]
        X[strDataset] = HighLevelFeaturesOnly(dfrSingleDataset).values
        y[strDataset] = dfrSingleDataset[strTarget].values
        astrPropulsion = dfrSingleDataset["propulsion"].values
        aintPropulsion = [dintCodes[strPropulsion] for strPropulsion in astrPropulsion]
        daintPropulsion[strDataset] = aintPropulsion
        if blnVerbose:
            print(f"- {strDataset}: {X[strDataset].shape} and {y[strDataset].shape}")
    if blnVerbose:
        print()
    
    # finalize
    return X,y,daintPropulsion      
    
import matplotlib
import matplotlib.patches as mpatches
def PlotPrediction(strTarget,y_true,y_pred,fltRmse,strDataset,varColor="orange",fltNoiseSize=-1,blnLogY=False):
    '''
    Plots a prediction table:
    - horizontal axis: true values.
    - vertical axis: predicted values.
    2022 03 10 dh Created
    2022 03 13 dh Tick limits reduced to current true/predicted values (i.e. not including train/valid any more)
    '''
    # define limits of axes
    if False: 
        # 2022 03 13: do not remember why I chose to include train/valid
        # afltMinMaxSource now removed from parameter list of the function (2nd position)
        afltAllTargetValues = np.concatenate([afltMinMaxSource, y_pred])  
    else: # 2022 03 13
        afltAllTargetValues = np.concatenate([y_true, y_pred])  
        
    fltMin = np.min(afltAllTargetValues)
    fltMax = np.max(afltAllTargetValues)         
                
    # add noise the separate dots
    if fltNoiseSize > 0:
        fltAxisExtensionFromNoise = u.AxisExtensionFromNoise(fltNoiseSize)
        fltMin -= fltAxisExtensionFromNoise
        fltMax += fltAxisExtensionFromNoise
        y_true = u.NoiseAdded(y_true,fltNoiseSize)
        y_pred = u.NoiseAdded(y_pred,fltNoiseSize)
    
    # RMSE as a factor
    strRmseAsFactor = ""
    if blnLogY:
        strRmseAsFactor = f" (as a factor: {round(10**fltRmse,3)})"
    else:
        strRmseAsFactor = ""
        
    # prepare plot
    fltSize = 5
    plt.figure(figsize=(fltSize, fltSize))
    objListedColormap = matplotlib.colors.ListedColormap(["red","blue","yellow"])

    # draw plot
    plt.scatter(y_true,y_pred, c=varColor, cmap=objListedColormap, edgecolor="black",alpha=0.5)
    plt.plot((fltMin,fltMax),(fltMin,fltMax), c="red") 
    plt.title(f"Predicting target '{strTarget}' from HLF\nRMSE = {round(fltRmse,3)}{strRmseAsFactor}")
    plt.xlabel(f"true '{strTarget}' of dataset '{strDataset}'")
    plt.ylabel(f"predicted '{strTarget}' of dataset '{strDataset}'")
    plt.xlim(fltMin,fltMax)
    plt.ylim(fltMin,fltMax)

    # artificial legend
    objPatchRed = mpatches.Patch(facecolor='red', label='steam', edgecolor="black")
    objPatchBlue = mpatches.Patch(facecolor='blue', label='electric', edgecolor="black")
    objPatchYellow = mpatches.Patch(facecolor='yellow', label='diesel', edgecolor="black")
    plt.legend(handles=[objPatchRed,objPatchBlue,objPatchYellow])

    # finalize
    plt.show();       

import pandas as pd    
def AnalyseWorstEstimates(dvarSummary, blnForceToInteger=False, blnBackTransform=False, intExamples = 5):
    '''
    Finds the locomotives with most extreme deviations from the true value:
    - extracts most extreme deviations, both too low and too high.
    - adds the identifier (i.e. column 'filenameroot')
    - looks up the images (Inception only)
    - plots the images
    From dvarSummary, only the following columns are used:
    - "Test y true"
    - "Test y predicted"
    2022 03 11 dh Created
    2022 03 13 Allow for back-transformation; rounding
    '''    
    # init
    cintMaxNameLength = 30
    
    # get data from dataset "test"
    y = {}
    y["true"] = dvarSummary["Test y true"]
    y["predicted"] = dvarSummary["Test y predicted"]
    
    # back-transform from log scale
    if blnBackTransform:
        for strYType in ["true","predicted"]:
            y[strYType] = 10 ** y[strYType]
            
    # calculate deviations in dataset "test"
    y["deviation"] = y["predicted"] - y["true"]                 
            
    # streamline true/predicted value for display
    for strYType in ["true","predicted","deviation"]:
        if blnForceToInteger: 
            y[strYType] = y[strYType].astype(int) # convert to integer
        else:
            y[strYType] = np.around(y[strYType]) # round       
            
    # add filename root (i.e. identifier)
    dfrScaled = u.FromDisk(f"dfrScaledMobileNet")
    astrFileNameRoots = dfrScaled[dfrScaled["dataset"] == "test"]["filenameroot"].values
    dfrTestPredictions = pd.DataFrame({
        "filenameroot":astrFileNameRoots,
        "Test y true":      y["true"],
        "Test y predicted": y["predicted"],
        "Test y deviation": y["deviation"]
    })
    
    # get images
    dimgInception = u.FromDisk("dimgInception")

    # find extremes on both sides
    for blnAscending in [False,True]:
        
        # init
        strSign = "" if blnAscending else "+"
        strDirection = "low" if blnAscending else "high"
        print(f"Estimate too {strDirection}:".upper())        
        dfrTestPredictions.sort_values("Test y deviation", ascending=blnAscending, inplace=True)
        lstrTitles = []
        limgPhotos = []
        
        # collect photos, including true values, predicted values and deviations
        for intExample in range(intExamples):
            [strFilenameRoot,fltTrue,fltPredicted,fltDeviation] = dfrTestPredictions.iloc[intExample,:4]
            strNameAbbr = u.Abbreviation(strFilenameRoot[5:],cintMaxNameLength)
            strTitle = f"{strNameAbbr}\n{fltTrue} {u.Symbol('arrow')} {fltPredicted} {u.Symbol('DELTA')}={strSign}{fltDeviation}"
            lstrTitles.append(strTitle)
            limgPhotos.append(dimgInception[strFilenameRoot])        
        
        # display photos and information on deviations
        u.PhotoGallery(limgPhotos, lstrTitles, intColumns=intExamples, fltWidth=20, fltHeight=3)  
        
from sklearn.metrics import confusion_matrix
def DisplayConfusionMatrix(y_true,y_pred,lstrCategories):
    '''
    Calculates and displays a confusion matrix.
    Used for post-modeling analysis after classification.
    2022 03 22 dh Created
    '''
    # calculate and turn into dataframe
    aintConfusionCounts = confusion_matrix(y_true, y_pred)
    dfrConfusionCounts = pd.DataFrame(aintConfusionCounts)
    
    # label columns and rows
    # - tag with "true" or "predicted" 
    dfrConfusionCounts.index   = [strCategory + " true"      for strCategory in lstrCategories]
    dfrConfusionCounts.columns = [strCategory + " predicted" for strCategory in lstrCategories]
    
    # display
    u.DisplayDataFrame(dfrConfusionCounts)      
    
import tensorflow as tf
import tensorflow_hub as hub
gcintChannels = 3
def SequentialModel (strModelName):
    '''
    Loads a transfer learning model.
    Current models available: Inception and MobileNet.
    2022 02 22 dh Created
    '''
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
        objSequentialModel = tf.keras.Sequential([hub.KerasLayer(strModelUrl, trainable=False)])  # can be True, see below.     
    except:
        print("The sequential model cannot be loaded.")
        print(r"Maybe the folder 'C:\Users\domin\AppData\Local\Temp\tfhub_modules' has to be deleted.")
        print(r"At least this was the case in ADS-ML course project 4, under certain conditions.")  
        return
    
    # build model
    lintBatchInputShape = [None, intSquareSide, intSquareSide, gcintChannels]
    objSequentialModel.build(lintBatchInputShape)
    print(f"The sequential model '{strModelName}' has been loaded successfully.")
    return objSequentialModel    
    
def SplitByColumnTest(dfrSource, varXColumns, strYColumn):
    '''
    Splits a dataframe by the binary entry in column "Test".
    
    When     Who What
    28.06.22 dh  Created
    '''
    
    # split
    dfrTrain = u.SingleFilter(dfrSource,"Test", "== 0")
    dfrTest  = u.SingleFilter(dfrSource,"Test", "== 1")    

    # allow for single strings and lists of strings
    if isinstance(varXColumns, str):
        varXColumns = [varXColumns]
    
    # turn into numpy arrays
    X = {}
    y = {}
    X["train"] = np.array(dfrTrain[varXColumns].values)
    y["train"] = np.array(dfrTrain[strYColumn].values)
    X["test"]  = np.array(dfrTest[varXColumns].values)
    y["test"]  = np.array(dfrTest[strYColumn].values)
    
    # finalize
    return X,y        