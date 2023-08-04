import pandas as pd
import numpy as np

def NarrowCast(varValue):   
    '''
    Tries to cast the most narrow way, i.e.: string > float > integer.
    2021 DH
    ''' 
    try:   
        if float(varValue) == int(varValue):
            varValue = int(varValue)
        else:
            varValue = float(varValue)
    except:
        try:    
            varValue = float(varValue)
        except:
            pass
    return varValue
if False:
    avarTesters = ["17Peter","17.1","17"]
    for varTester in avarTesters:
        print(varTester, "yields", type(NarrowCast(varTester)))
        
def StripStringAfter(strHaystack, strNeedle):
    '''
    Strips anything after strNeedle from strHaystack, including strNeedle
    2021 DH
    '''
    return strHaystack.split(strNeedle, 1)[0]
if False:
    print (StripStringAfter("tsrX) # Peter",")"))  
    
import inspect      
def VariableName(a):
    ''' 
    Returns the name of a variable.
    Programmed by Ivo Wetzel.
    Major re-design: DH.
    2021 10 17: DH: shape added.
    '''
    fraCurrentFrame = inspect.currentframe()
    fraCurrentFrame = inspect.getouterframes(fraCurrentFrame)[2] # DH: adjustment of index
    strCodeContext = inspect.getframeinfo(fraCurrentFrame[0]).code_context[0].strip()
    strArgument = strCodeContext[strCodeContext.find('(') + 1:-1].split(',')[0]
    if strArgument.find('=') != -1:
        strResult = strArgument.split('=')[1].strip()
    else:
        strResult = strArgument
    return strResult

def TypeChecker(varSource, intSampleLength = 50):
    '''
    Prints the type of a variable or object.
    varSource:       variable or object to be analyzed
    intSampleLength: abbreviation cut-off, default: 50
    2021: DH
    '''
    y = varSource
    strVariableName = VariableName(y)
    strVariableName = StripStringAfter(strVariableName,")") # sometimes a disturbing ")..." is added
    strContents = str(varSource)
    if len(strContents) > intSampleLength:
        strContents = f"{strContents[:intSampleLength]}..." 
    
    # shape
    try:
      strShape = f"Shape: {varSource.shape}"
    except:
      try:
        strShape = f"Length: {len(varSource)}"    	
      except:
        strShape = "No shape or length available."
        
    print(f"\nTYPE-CHECKER:",
        f"Name: {strVariableName}",
        f"Contents: {strContents}",
        f"Type:     {type(varSource)}",
        strShape, 
        "\n", sep="\n")
        
if False:
    lstrFunnyList=[17,True,"Peter"]
    t(lstrFunnyList)
    
def VariableToDisk(varValue):
    '''
    Writes value of a variable into a file with the name of the variable plus extension txt.
    2021: DH.
    '''
    strVariableName = VariableName(varValue)
    VariableOnDisk(varValue, strVariableName, "set")

def VariableFromDisk(strVariableName):
    '''
    Retrieves the value of a variable from disk.
    2021: DH.
    '''
    varVariableValue = VariableOnDisk(None, strVariableName, "get")
    return varVariableValue
    
def VariableOnDisk(varValue, strVariableName, strDirection): 
    '''
    Writes and reads variable values from disk.
    2021: DH.
    '''
    if strDirection=="set":
        objFile = open(f"{strVariableName}.txt", "w")
        objFile.write(str(varValue))
    elif strDirection=="get":
        objFile = open(f"{strVariableName}.txt", "r")
        varResult = objFile.read()       
    objFile.close
    if strDirection=="get":
        return NarrowCast(varResult)
					
if False:
    strMyCar = "Bentley"
    VariableToDisk(strMyCar)
    strMyCar = "cleaned"
    strMyCar = VariableFromDisk("strMyCar")
    print(strMyCar) 

import matplotlib.pyplot as plt
import numpy as np
def Roots(afltX, afltY):
    '''
    Calculates the root(s) of a function.
    2021: DH.
    '''
    lfltRoots = []
    for intCurrPoint in range(len(afltX)-1):
        fltCurrX = afltX[intCurrPoint]
        fltCurrY = afltY[intCurrPoint]
        fltNextX = afltX[intCurrPoint+1]
        fltNextY = afltY[intCurrPoint+1]
        if np.sign(fltCurrY) != np.sign(fltNextY):
            fltRoot = fltCurrX - fltCurrY * (fltNextX - fltCurrX) / (fltNextY - fltCurrY)
            lfltRoots.append(fltRoot)  
    lfltRoots = list(set(lfltRoots)) # remove duplicates
    lfltRoots.sort()

    if len(lfltRoots) == 0:
        strRootInformation = "No roots."
    else:
        lstrRoots = []
        strRootPluralSingular = "Root"
        if len(lfltRoots) > 1:
            strRootPluralSingular = f"{strRootPluralSingular}s"
        for fltRoot in lfltRoots:
            lstrRoots.append(str(round(fltRoot,2)))
        strRootInformation = ', '.join(lstrRoots)
        strRootInformation = f"{strRootPluralSingular}: {strRootInformation}"
    return np.array(lfltRoots), strRootInformation

def Derivative(afltX, afltY):
    '''
    Calculates the derivative of a function.
    2021: DH.
    '''
    lfltNewX = []
    lfltNewY = []
    for intCurrPoint in range(len(afltX)-1):
        fltCurrX = afltX[intCurrPoint]
        fltCurrY = afltY[intCurrPoint]
        fltNextX = afltX[intCurrPoint+1]
        fltNextY = afltY[intCurrPoint+1]
        fltMeanX = np.mean([fltCurrX,fltNextX])
        fltDerivative = (fltNextY-fltCurrY) / (fltNextX-fltCurrX)
        lfltNewX.append(fltMeanX)   
        lfltNewY.append(fltDerivative)   
    return np.array(lfltNewX),np.array(lfltNewY)
def PlotFunction(afltX, afltY, strInfo = ""):
    '''
    Creates a simple line plot, high-lighting the x axis.
    2021: DH.
    '''
    plt.figure(figsize=(6,2))
    plt.plot(afltX, afltY)
    plt.plot([np.min(afltX),np.max(afltX)], [0,0])
    plt.title(strInfo)
    plt.show
    
def PlotDerivatives(afltX, afltY, intDepth = 1):
    '''
    Plots one or several derivates of a function.
    2021: DH.
    '''
    afltRoots, strRootInformation = Roots(afltX, afltY)            
    PlotFunction(afltX, afltY, f"Original function\n{strRootInformation}")
    for intRepetition in range(intDepth):
        (afltX, afltY) = Derivative(afltX, afltY)
        afltRoots, strRootInformation = Roots(afltX, afltY)    
        PlotFunction(afltX, afltY, f"Derivative {intRepetition+1}\n{strRootInformation}")
    return afltRoots
if False: 
    afltY = np.array([5, 20, 29, 37, 43.5, 50,55,59,60])
    afltX = np.array(range(len(afltY))) * 10
    afltRoots = PlotDerivatives(afltX,afltY,3)
    print("The roots of the last derivative are:", list(afltRoots))
    
import re
def SplitCamelCase(strSource, blnUpperCaseFirstOnly = True, strSeparator=" "):
    '''
    Credits: Matthias from StackOverflow.
    https://stackoverflow.com/questions/5020906/python-convert-camel-case-to-space-delimited-using-regex-and-taking-acronyms-in#9283563
    2021: Adjusted by DH.
    '''
    strResult = re.sub(r'((?<=[a-z])[A-Z]|(?<!\A)[A-Z](?=[a-z]))', r' \1', strSource)
    if strSeparator != "":
    	strResult = strResult.replace(" ", strSeparator)
    if blnUpperCaseFirstOnly:
    	strResult = strResult.capitalize() 
    	
    return strResult
if False:
    print(SplitCamelCase("MyCamelCase"))   
    
def DisplayDataFrame(dfrSource):
		'''
		Used to display a dataframe within a cell, i.e. not at the end of a cell
		2021: DH.
		'''
		from IPython.display import display
		display(dfrSource)
    
import pandas as pd
def SingleFilter(dfrSource, strColumn:str, strCondition:str):
    '''
    Returns a dataframe filtered by a single condition.
    2021: DH.
    '''
    return eval(f"dfrSource[dfrSource['{strColumn}'] {strCondition}]")
    
def PlotPhoto(aintFlatImage, intHeight, intWidth, blnColor, strTitle="", fltSize=3., blnDropTicks = False):
    '''
    Converts a flat pixel array into a photo plot.
    2021 12 XX dh Created
    2022 03 22 dh Allow for dropping ticks
    '''
    
    # reshape
    if blnColor:
        aintImage0Reshaped = aintFlatImage.reshape(intHeight, intWidth, 3) 
    else:
        aintImage0Reshaped = aintFlatImage.reshape(intHeight, intWidth ) 

    # plot the image
    plt.figure(figsize=(fltSize,fltSize))
    ax = plt.imshow(aintImage0Reshaped)
    if blnDropTicks:
	    ax.axes.xaxis.set_visible(False)
	    ax.axes.yaxis.set_visible(False)
    plt.title(strTitle)
    plt.show()

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd

def SurfacePlot(srsX, srsY, srsZ, fltSize=11, fltElevation=5, fltAngle=45):
    '''
    Creates a surface plot.
    Inspired by https://www.python-graph-gallery.com/342-animation-on-3d-plot
    2021: Adjusted by DH.
    '''
    lfltX = list(srsX)
    lfltY = list(srsY)
    lfltZ = list(srsZ)

    # plot    
    fig = plt.figure(figsize=(fltSize,fltSize))

    ax = fig.gca(projection='3d')
    ax.plot_trisurf(lfltX, lfltY, lfltZ, cmap=plt.cm.viridis, linewidth=0.2)
    plt.xlabel(srsX.name)
    plt.ylabel(srsY.name)
    plt.title(srsZ.name)

    # set the angle of the camera
    ax.view_init(fltElevation, fltAngle)
    
if False:
    dfrTest = pd.DataFrame({
        "alpha": [1,2,3,1,2,3,1,2,3],
        "beta": [1,1,1,2,2,2,3,3,3],
        "gamma": [11,12,3,2,4,6,3,6,9]
    })
    SurfacePlot(dfrTest["alpha"],dfrTest["beta"],dfrTest["gamma"], 12, 45, 45)
    
def AsPercentage(fltValue):
		'''
		Converts a float number into a rounded percentage value, including a percentage sign.
		2021: DH.
		'''
		return str(round(fltValue*100,2)) + "%"
    
def SingularPlural(intCount, strSingular, strPlural=""):
    '''
    Returns proper singular or plural, depending on the count.
    2021: DH.
    '''
    if strPlural == "":
        strPlural = f"{strSingular}s"
    if intCount == 1:
        return f"{intCount} {strSingular}"
    else:
        return f"{intCount} {strPlural}"
if False:
    for intCount in range(3):
        print(SingularPlural(intCount,"house"))
        print(SingularPlural(intCount,"woman","women"))      
        

import matplotlib.pyplot as plt
import math as math
def PhotoGallery(aimgPhotos,lstrTitles,intColumns,fltWidth,fltHeight):
    '''
    Prints photos as a gallery.
    Instaed of an array aimgPhotos there can alos be a list: limgPhotos
    Extends PlotPhoto().
    2021       dh Created
    2022 02 09 dh Color map defined explicitely
    '''
    
    # init   
    intImageCount = len(aimgPhotos)
    intRows = math.ceil(intImageCount / intColumns)
    objFigure, aintAxes = plt.subplots(nrows=intRows, ncols=intColumns, figsize=(fltWidth, fltHeight))

    if len(aintAxes.shape) == 1:
        aintAxes = aintAxes[np.newaxis, :]
    intCurrImage = 0
    
    # plot each image at the right row/column position 
    for intRowPointer, aintAxis in enumerate(aintAxes):
        for intColPointer, aspAxesSubplot in enumerate(aintAxis):

            try:
                # get photo and show it
                imgPhoto = aimgPhotos[intCurrImage]
                strTitle = lstrTitles[intCurrImage]
                aspAxesSubplot.imshow(imgPhoto, cmap=plt.cm.gray) # default cmap is "viridis"
                aspAxesSubplot.get_xaxis().set_visible(False) # hide ticks
                aspAxesSubplot.get_yaxis().set_visible(False)
                aspAxesSubplot.set_title(strTitle)
            except:
                # suppress the empty image frame
                aspAxesSubplot.remove()
            
            # prepare next image
            intCurrImage += 1

    # finalize
    plt.show();

if False:
    PhotoGallery(avarGalleryImages,lstrTitles,5,15,5) 
    
def DataFrameColumnMoved (dfrSource, strColumn, intTargetPosition = 0):
    '''
    Moves a dataframe column to a new position.
    Default: to the left end.   
    2021 12 31: DH.
    '''
    srsColumnToMove = dfrSource.pop(strColumn)
    dfrSource.insert(0, strColumn, srsColumnToMove)
    return dfrSource
    
def DoesContainDigits(strSource):
    '''
    Checks if a string contains digits.
    2022 01 01: DH
    '''    
    return bool(re.search(r'\d', strSource))
if False:
    lstrTesters = ["Peter","PeterPaul","LNC","17","afltX"]
    for strTester in lstrTesters:
        print(strTester,DoesContainDigits(strTester))
        
def DoesContainUnderscore(strSource):
    '''
    Checks if a string contains an underscore.
    2022 01 01: DH
    '''    
    return bool(re.search("_", strSource))
if False:
    lstrTesters = ["Peter","Peter_Paul","LNC_","17","afltX"]
    for strTester in lstrTesters:
        print(strTester,DoesContainUnderscore(strTester))
        
def IsCamelCase(strSource):
    '''
    Checks if a string is written in camel-case.
    2022 01 01: DH
    '''
    if strSource == strSource.upper():
        return False
    elif strSource[1:] == strSource[1:].lower():
        return False
    else:
        return True
if False:
    lstrTesters = ["Peter","PeterPaul","LNC","17","afltX"]
    for strTester in lstrTesters:
        print(strTester,IsCamelCase(strTester))    

def CharsDropped (strHaystack, strNeedles):
    '''
    Drops the characters in strNeedles from strHaystack.
    Could also be solved with regexp.
    12.01.22 DH Created.
    '''    
    for strNeedle in strNeedles:
        strHaystack = strHaystack.replace(strNeedle, "")
    return strHaystack
    
def CharsTranslated (strSource, strOldChars, strNewChars):
    '''
    Translates some characters.
    Other solution: standard string function translate().
    12.01.22 DH Created.
    '''    
    if len(strOldChars) == len(strNewChars):
        for intPointer, strOldChar in enumerate(strOldChars):
            strNewChar = strNewChars[intPointer]
            strSource = strSource.replace(strOldChar,strNewChar)
    else:
        print (f"Warning: the strings '{strOldChars}' and '{strNewChars}' must have the same length in function CharsTranslated.")

    # finalize
    return strSource

if False:
    print(1, CharsTranslated("This should be changed.", "aeiou", "AEIOU"))
    print(2, CharsTranslated("This should be changed.", "aeiou", "AEIO"))         
        
def TranslationsByDictionary(strSource, dstrDictionary):
    '''
    Replaces some text chunks in a string, according to a dictionary.
    2022 01 15 DH Created
    '''
    strResult = strSource
    for strOld, strNew in dstrDictionary.items():
        strResult = strResult.replace(strOld, strNew)
    return strResult

if False:
    dstrTranslations = {
        "&nbsp;":" ", # non-breaking space
        ",<br>" :",", # line break
        "<br>"  :","  # line break
    }    
    print(TranslationsByDictionary("Peter,<br>Paul",dstrTranslations))     
    
import time; ms = time.time()*1000.0
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
if False:
    StopWatch()        
    for i in range(10000000):
        x=2.75**25
    print(f"We have to wait for {StopWatch('sec',2,False)}.")   
    
def Extension(strFilename):
    '''
    Extracts the file extension of a filename.
    '''
    return strFilename.split(".")[-1]
def WithoutExtension(strFilename):
    '''
    Returns a filename without its extension and without the dot before the extension.
    '''
    intCharsToCut = len(Extension(strFilename))+1
    return strFilename[:-intCharsToCut]
if False:
    print(WithoutExtension("Peter.und.Paul.jpg"))     
    
def RemoveParentheses(strSource):
    '''
    Drops parentheses and its contents from a string.
    Adjusts for extra spaces.
    2022 01 28 DH Created
    '''
    strResult = re.sub(r"\((.*?)\)", "", strSource)
    strResult = strResult.replace("  "," ")
    strResult = strResult.strip()
    return strResult
if False:
    lstrTesters = ["Peter (Paul)", "Peter (und) Paul", "(Peter)"]
    for strTester in lstrTesters:
        print(f"{strTester} --> XXX{RemoveParentheses(strTester)}XXX")    
        
import pickle
gcstrPicklePath = "./PickleFiles/"
def ToDisk(varObject,strForcedName="",strType=""):
	'''
	Writes a variable or an object into a "pickle" file.
	Target folder: gcstrPicklePath.
	Filename: the name of the variable or object plus extension "p".
	This function is an extension of VariableToDisk(varValue)    
	2022 01 31: dh Created
	2023 03 14: dh Allowing for a fixed name
	2023 06 20: dh Adjusted to Cookiecutter
	2023 07 02: dh Adjusted to Cookiecutter
	2023 08 02: dh Allow for paths in the forced name
	'''    
	# define path
	if strType.lower() in ["models","model","m"]:
		strPath = "../models/"	
	elif strType.lower() in ["processed","p"]:
		strPath = "../data/processed/"	
	elif "/" in strForcedName:
		lstrParts = strForcedName.rsplit('/', 1)
		strPath = lstrParts[0] + '/'
		strForcedName = lstrParts[1]
	else:
		strPath = gcstrPicklePath  
	  
	# define full filename
	if strForcedName == "":
		strObjectName = VariableName(varObject)		
	else:
		strObjectName = strForcedName
	strObjectName = strObjectName.strip()
	strFilename = f"{strPath}{strObjectName}.p"
	
	# save	
	# wb = write in binary mode
	with open(strFilename, 'wb') as objBufferedWriter: 
	  pickle.dump(varObject, objBufferedWriter)
        
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
    
if False:
    import pandas as pd
    print("Dictionary".upper())
    dlfltExample = {
        'x': [6.28318, 2.71828, 1],
        'y': [2, 3, 5]
    }
    print("BEFORE:", dlfltExample)
    ToDisk(dlfltExample)
    dlfltExample = FromDisk("dlfltExample")
    print("AFTER: ", dlfltExample)
    print()
    
    print("Dataframe".upper())
    dfrExample = pd.DataFrame(dlfltExample)
    print("BEFORE:")
    DisplayDataFrame(dfrExample)  
    ToDisk(dfrExample)
    dfrExample = FromDisk("dfrExample")
    print("AFTER: ")
    DisplayDataFrame(dfrExample)   
    print()
    
    print("Float".upper())
    fltExample = 17.18
    print("BEFORE:", fltExample)
    ToDisk(fltExample)
    fltExample = FromDisk("fltExample")
    print("AFTER: ", fltExample)
    print()    
    
def RemoveBlanksFromList(lvarElements):
    '''
    Removes blank strings from a list.
    The method remove() is not useful:
    - error if no blank found.
    - removes only 1 instance
    2022 01 31 DH Created
    '''
    return [varElement for varElement in lvarElements if varElement != ""]
if False:
    lstrExamples = [""] * 7 + ["Peter"] * 2 + ["Paul"] * 2 
    print(lstrExamples)
    RemoveBlanksFromList(lstrExamples)
    lstrExamples = RemoveBlanksFromList(lstrExamples)
    print(lstrExamples)        
    
def StripList(lstrElements):
    '''
    Removes leading and trailing blanks from list elements.
    2022 01 31 DH Created
    '''
    return [strElement.strip() for strElement in lstrElements]
if False:
    lstrExamples = ["  Peter", "   Paul   "]
    print(lstrExamples)
    lstrExamples = StripList(lstrExamples)
    print(lstrExamples)    
    
def DropHumanNanExpressions(varSource):
    '''
    Drops human expressions that indicate NaN.
    2022 01 31 DH Created
    '''
    if isinstance(varSource,str):
        lstrVerboseNan = ["&nbsp;","?","(?)","(None)","None","(none)","none","(Unknown)","Unknown","(unknown)","unknown","-","()"]
        for strVerboseNan in lstrVerboseNan:
            varSource = varSource.replace(strVerboseNan, "")
    return varSource
if False:
    print("-",DropHumanNanExpressions("(None)"),"-", sep="") 
    
import pandas as pd
import re
def SeriesWithoutHtmlTags(srsColumn):
    '''
    Removes HTML tags from a dataframe column (i.e. a series).
    Inspired by: https://stackoverflow.com/questions/50447559/apply-html-tags-removal-to-pandas-column
    2022 02 02 DH Created
    '''
    for intPointer,varValue in enumerate(srsColumn):
        if isinstance(varValue,str):
            varValue = re.sub('<[^<]+?>', '', varValue)
            srsColumn.loc[intPointer] = varValue
    return srsColumn
if False:
    dfrTester = pd.DataFrame({"description": ['<p>Hello</p>', '<p>World</p>',17]})
    print(dfrTester)
    dfrTester["description"] = SeriesWithoutHtmlTags(dfrTester["description"])
    print(dfrTester)                
    
def PrintAlias(*tvarArguments, sep=" "):
    '''
    Abbreviation for print().
    To be imported using: "from Utilities import PrintAlias as p"
    2022 02 10 dh Created
    '''
    strResult = ""
    for varArgument in tvarArguments:
        strArgument = f"{varArgument}"
        if strResult == "":
            strResult = strArgument
        else:
            strResult = f"{strResult}{sep}{strArgument}"
    print(strResult)

if False:
    PrintAlias(15,"alpha","beta",17,f"{19*19}", sep="---|---")
    
import pandas as pd
def SetExtremeValuesToNan(dfrSource,strColumn,fltLow, fltHigh):
    '''
    Sets extreme values of a dataframe column to NaN.
    2022 02 10 dh Created
    '''
    try:
        dfrSource[strColumn] = dfrSource[strColumn].where((dfrSource[strColumn]>=fltLow) & (dfrSource[strColumn]<=fltHigh))
    except:
        print(f"Warning: there are non-numeric values in column {strColumn}.")
if False:    
    df = pd.DataFrame({'A': ["peter", 1, 3, 2], 'B': ['red', 'white', 'blue', 'green']})
    print(df)
    SetExtremeValuesToNan(df,"A",1,3)
    print(df) 
      
def CategoryCounts (srsSource,intLimit=-1):
    '''
    Displays counts of a series, both absolute and relative.
    Returns the number of categories.
    2022 02 14 dh Created
    '''
    srsAbsolute = srsSource.value_counts(normalize=False)
    srsRelative = srsSource.value_counts(normalize=True)
    srsRelative = round(srsRelative * 100,1).astype(str) + "%"
    dfrCounts = pd.concat([srsAbsolute,srsRelative],axis=1)
    dfrCounts.columns = ["absolute","relative"]
    if intLimit == -1:
    	DisplayDataFrame(dfrCounts)     
    else:
    	DisplayDataFrame(dfrCounts.head(intLimit))     
    return dfrCounts.shape[0]
    
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
def HeatMap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", fltDegreesTickLabelRotation = 0, **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
        
    Source: https://matplotlib.org/3.4.3/gallery/images_contours_and_fields/image_annotated_heatmap.html#sphx-glr-gallery-images-contours-and-fields-image-annotated-heatmap-py    
    
    2022 02 24 dh Copied from MatPlotLib
    2022 02 24 dh Function name adjusted
    2022 02 24 dh Degrees tick label rotation: default to 0
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=fltDegreesTickLabelRotation, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def AnnotateHeatMap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
        
    Source: https://matplotlib.org/3.4.3/gallery/images_contours_and_fields/image_annotated_heatmap.html#sphx-glr-gallery-images-contours-and-fields-image-annotated-heatmap-py    
    
    2022 02 24 dh Copied from MatPlotLib       
    2022 02 24 dh Function name adjusted        
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

def AnnotatedHeatMap(afltTableValues, lstrRowLabels, lstrColLabels, 
             strColorBarLabel="", fltWidth=20, fltHeight=7,strValueFormat = "{x:.0f}", fltDegreesTickLabelRotation=0, strColorMap = "Oranges", **kwargs):
    '''
    Wraps HeatMap() and AnnotateHeatMap(): displays an annotated heatmap.
    Example for strValueFormat: "{x:.3f} kg"
    Color maps see https://matplotlib.org/stable/tutorials/colors/colormaps.html
    2022 02 24 dh Created    
    '''
    objFigure, objAxesSubplot = plt.subplots(figsize=(fltWidth,fltHeight))

    objAxesImage, objColorBar = HeatMap(
        afltTableValues, lstrRowLabels, lstrColLabels, ax=objAxesSubplot,
        cmap=strColorMap, cbarlabel=strColorBarLabel, fltDegreesTickLabelRotation=fltDegreesTickLabelRotation
    )
    lobjMatPlotLibTexts = AnnotateHeatMap(objAxesImage, valfmt=strValueFormat)
    objFigure.tight_layout()
    plt.show()    

if False:
    # source: https://matplotlib.org/3.4.3/gallery/images_contours_and_fields/image_annotated_heatmap.html#sphx-glr-gallery-images-contours-and-fields-image-annotated-heatmap-py
    lstrVegetables = ["cucumber", "tomato", "lettuce", "asparagus",
                  "potato", "wheat", "barley"]
    lstrFarmers = ["Farmer Joe", "Upland Bros.", "Smith Gardening",
               "Agrifun", "Organiculture", "BioGoods Ltd.", "Cornylee Corp."]
    afltHarvest = np.array([[0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
                        [2.4, 0.0, 4.0, 1.0, 2.7, 0.0, 0.0],
                        [1.1, 2.4, 0.8, 4.3, 1.9, 4.4, 0.0],
                        [0.6, 0.0, 0.3, 0.0, 3.1, 0.0, 0.0],
                        [0.7, 1.7, 0.6, 2.6, 2.2, 6.2, 0.0],
                        [1.3, 1.2, 0.0, 0.0, 0.0, 3.2, 5.1],
                        [0.1, 2.0, 0.0, 1.4, 0.0, 1.9, 6.3]])
    AnnotatedHeatMap(afltHarvest, lstrVegetables, lstrFarmers, "harvest [t/year]", 10,5, fltDegreesTickLabelRotation=-45, strColorMap="OrRd")

import numpy as np
def Log10Plus1 (fltValue):
    '''
    Returns logarithm on base 10.
    Shifts source value by 1 unit to avoid problems with 0.
    2022 02 25 dh Created
    '''
    return np.log10(fltValue+1)
if False:
    p(Log10Plus1(0))

def Log10Plus1Inverse (fltValue):
    '''
    Reverses Log10Plus1.
    2022 03 12 dh Created
    '''
    return (10 ** fltValue) - 1
if False:
    fltTester = 123
    x = Log10Plus1(fltTester)
    print(x)
    x = Log10Plus1Inverse(x)
    print(x)
    
import numpy as np
def JoinAnyType(lvarSource, strSeparator = ", "):
    '''
    Joins any elements from lists and Numpy array, not only strings.
    2022 03 10 dh Created
    '''
    lstrSource = list(map(str, lvarSource))
    return strSeparator.join(lstrSource)
if False:
    lvarAsList = ["Peter",17,17.17]
    avarAsArray = np.array(lvarAsList)
    lvarTesters = [lvarAsList,avarAsArray]
    for varTester in lvarTesters:
        print(JoinAnyType(varTester))    
        
def Abbreviation(strSource,intLength = 12):
    '''
    Returns abbreviation of a string.
    2022 03 11 dh Created
    '''
    if len(strSource) <= intLength:
        return strSource
    else:
        return strSource[:intLength-3] + "..."
if False:
    lstrTesters = ["Alpha","Alphasaurus lophogaster"]
    for strTester in lstrTesters:
        print(Abbreviation(strTester),Abbreviation(strTester,7))   
        
def FunctionName(objFunction):
    '''
    Returns the name of a function. 
    Example for "str(objFunction)": "<function Log10Plus1 at 0x000001D108270160>"
    2022 03 13 dh Created
    '''
    lstrWords = str(objFunction).split() # this returns list of words
    strFunctionName = lstrWords[1]
    strFunctionName = CharsDropped (strFunctionName, "'<>")
    return(strFunctionName)
if False:
    print(FunctionName(u.Log10Plus1))
    import numpy as np
    print(FunctionName(np.log10))   
    
import numpy as np
def NoiseAdded(afltSource, fltNoiseSize):
    '''
    Adds noise to an array.
    2022 03 11 dh Created
    '''
    afltNoise = np.random.uniform(-0.5, 0.5, size=len(afltSource)) * fltNoiseSize
    return afltSource + afltNoise
if False:
    afltSource = np.array([1,2,3,4,5])
    afltSource = NoiseAdded(afltSource,0.1)
    print(afltSource)                
    
import math
def AxisExtensionFromNoise(fltNoiseSize):
    '''
    Returns adequate axis extension:
    - Noise around integers requires extending the axes.
    - This extension depends on the noise size.
    2022 03 13 dh Created
    '''
    return 10 ** math.ceil(np.log10(fltNoiseSize))
if False:
    lfltTesters = [0.9,0.09,0.009]
    for fltTester in lfltTesters:
        print(f"{fltTester} --> {AxisExtensionFromNoise(fltTester)}")     
        
def Symbol(strSymbolName):
    '''
    Converts symbol name into the symbol itself.
    UTF-8 characters work in Jypiter notebooks, but not in outsourced Python libraries.
    2022 03 13 dh Created
    '''
    dintTranslations = {
        "arrow":8594,
        "alpha":945,
        "delta":948,
        "DELTA":916,
        "epsilon":949,
        "plusminus":177
    }
    return chr(dintTranslations[strSymbolName])
if False:
    print(ord("a")) # to define new characters
    print(Symbol("alpha"),Symbol("arrow"),Symbol("epsilon"))        
    
import operator
def SortDictionaryByValue(dfltSource, varAscending=True):
    '''
    Sorts a dictionary by its values
    Inspiration: https://www.w3resource.com/python-exercises/dictionary/python-data-type-dictionary-exercise-1.php
    2022 03 14 dh Created
    '''
    if str(varAscending).lower() in ["asc","ascending"]:
    	varAscending=True
    if str(varAscending).lower() in ["desc","descending"]:
    	varAscending=False
    	
    try:
        return dict(sorted(dfltSource.items(), key=operator.itemgetter(1),reverse=not varAscending))
    except:
        print("WARNING: In SortDictionaryByValue(), dictionary entries must be float or integer.")
if False:
    dfltTester = {"alpha":1,"beta":2,"gamma":3.3}
    p(SortDictionaryByValue(dfltTester,False))
    dfltTester = {"alpha":1,"beta":2.2,"gamma":(3,4)}
    p(SortDictionaryByValue(dfltTester,False))     
    
import numpy as np
def TranslateCategoriesToInteger(astrSource,lstrCategories):
    '''
    Translates an array of strings into an array of integers, according to categories in a fixed order.
    Inspiration: https://stackoverflow.com/questions/16992713/translate-every-element-in-numpy-array-according-to-key
    2022 03 19 dh Created
    '''
    dintTranslationTable = {}
    for intIndex,strCategory in enumerate(lstrCategories):
        dintTranslationTable[strCategory] = intIndex
    return np.vectorize(dintTranslationTable.get)(astrSource)
        
if False:
    astrTester = np.array(["alpha","beta","gamma","beta","alpha","beta","gamma","beta","alpha","beta","gamma","beta"])
    lstrCategories = ["alpha","beta","gamma"]
    astrTranslated = TranslateCategoriesToInteger(astrTester,lstrCategories)
    print(astrTranslated)       
    
def TargetCategories(strTarget):
    '''
    Returns categories of a nominal target as a list.
    2022 03 20 dh Created
    '''
    strTarget = strTarget.lower()
    if strTarget == "propulsion":
        lstrCategories = ["steam","electric","diesel"]
    elif strTarget == "axlecategory":
        lstrCategories = ['short','C',
                          '4 axles','BB','2B',
                          '5 axles','2C',
                          '6 axles','CC','2C1',
                          'long']
    else:
        print(f"WARNING: unknown target '{strTarget}' to translate categories into integers.")  
    return lstrCategories    
    
def UndefinedArticle(strNoun):
    '''
    Builds the English undefined article from the noun.
    In some cases, the string returned may be wrong.
    2022 03 23 dh Created
    '''
    if strNoun[0].lower()  in "aeio":
        return "an"
    elif strNoun[:2].lower()  == "un":
         return "an"
    else:       
        return "a"
if False:
    lstrTesters = ["house","elephant","American","user","undefined article"]
    for strTester in lstrTesters:
            p(UndefinedArticle(strTester),strTester)    

import os
import shutil
def EmptyFolder(strFolder):
    '''
    Empties a folder.
    "unlink" means: delete.
    "shutil" means: "shell utilities.
    Inspiration: https://stackoverflow.com/questions/185936/how-to-delete-the-contents-of-a-folder
    2022 03 25 dh Created
    '''
    lstrFilenames = os.listdir(path=strFolder)   
    for strFilename in lstrFilenames:
        strFilePath = os.path.join(strFolder, strFilename)
        try:
            if os.path.isfile(strFilePath) or os.path.islink(strFilePath):
                os.unlink(strFilePath)
            elif os.path.isdir(strFilePath):
                shutil.rmtree(strFilePath)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (strFilePath, e)) 

def SecondsToText(fltSeconds):
    '''
    Creates simple, easy-to-read time string in minutes or seconds.
    Used for progress information.
    2021 11 02 dh Created
    '''
    if fltSeconds > 120: # i.e. > 2 minutes
        intMinutes = round(fltSeconds / 60)
        strTime = f"{intMinutes} minutes"
    elif fltSeconds < 1:
        strTime = "not known yet"
    else:
        strTime = f"{round(fltSeconds)} seconds"
    return strTime                
    
def Decade(intYear):
    '''
    Derives decade from year.
    2022 03 28 dh Created
    '''
    return int(intYear / 10) * 10
if False:
    lintTesters = list(range(1960,1972))
    for intTester in lintTesters:
        print(intTester, Decade(intTester))   

def DictionarySortedByValues(dfltSource):
	'''
	Sorts a dictionary by its values.
	
	Inspired by https://stackoverflow.com/questions/613183/how-do-i-sort-a-dictionary-by-value.
	
	When     Who What
	01.07.22 dh  Created
	'''
	return {k: v for k, v in sorted(dfltSource.items(), key=lambda item: item[1])} 
		
import numpy as np
def LogarithmicList(intLower, intUpper, intSteps = 1):
    '''
    Generates a list of logarithmically spaced values.
    When     Who What
    04.07.22 dh  Created
    '''
    lfltResult = np.logspace(intLower, intUpper, (intUpper - intLower) * intSteps + 1, endpoint=True)
    return lfltResult
if False:
    intSteps = 2
    print(LogarithmicList(-3,2,intSteps))		         

import matplotlib.pyplot as plt
import numpy as np
import warnings

def CreateBoxPlot(dafltValues):
    '''
    Creates a box-plot from a dataframe.
    
    Expects a dictionary:
    - dictionary key used for label
    - values as an array (or a list)
    
    Whiskers: 1.5 times interquartile distance.
    
    Inspired by: https://www.geeksforgeeks.org/box-plot-in-python-using-matplotlib/
    
    When     Who What
    31.07.22 ch  Created
    '''
    
    # turn dictionary into list of arrays
    lafltValues = []
    lstrKeys = dafltValues.keys()
    for strKey in lstrKeys:
        lafltValues.append(dafltValues[strKey])

    # init
    fltWidth = 5
    fltHeight = 3
    fig = plt.figure(figsize =(fltWidth, fltHeight))

    # create axes instance
    ax = fig.add_axes([0, 0, 1, 1]) # fractions of left, bottom, width, height
    warnings.simplefilter("ignore") # avoid warning on FixedLocator
    ax.set_xticklabels(lstrKeys)
    warnings.simplefilter("default")

    # creating plot
    dlobjVisualizationElements = ax.boxplot(lafltValues)
    plt.show()

if False:
    aflt1 = np.random.normal(100, 10, 200)
    aflt2= np.random.normal(90, 20, 200)
    aflt3= np.random.normal(80, 30, 200)
    data = [aflt1,aflt2,aflt3]  
    dafltValues = {"one":aflt1,"two": aflt2,"three": aflt3}  
    CreateBoxPlot(dafltValues)     
    
def ArrayDifferenceCount(avar1, avar2):
    '''
    Counts the number of differences in an array.
    When     Who What
    17.08.22 dh  Created
    '''
    if len(avar1) != len(avar2):
        print("In ArrayDifferenceCount(), the arrays must have the same length.")
        return
    intCount = 0;
    for intElement in range(len(avar1)):
        if avar1[intElement] != avar2[intElement]:
            intCount += 1
    return intCount
if False:
    avar1 = np.array([1,2,3,4,5])
    avar2 = np.array([1,2,0,0,5])
    print(ArrayDifferenceCount(avar1, avar2))    
    
import random
import pandas as pd
def BestWords(lstrCurrWords,lfltProbabilities,intCount):
    '''
    Returns the best words at the current position, according to bigram probabilities.
    When     Who What
    06.09.22 dh  Created
    '''    
    dfrCurrProb = pd.DataFrame({"Words":lstrCurrWords,"Probabilities":lfltProbabilities})
    dfrCurrProb.sort_values("Probabilities", axis=0, ascending=False, inplace=True)
    lstrBestWords = list(dfrCurrProb["Words"].values)[:intCount]
    return lstrBestWords

def AnyWords(lstrWords,intCount):
    '''
    Returns a few words from a list at random.
    When     Who What
    06.09.22 dh  Created
    '''
    lstrWordsCopy = lstrWords.copy()
    random.shuffle(lstrWordsCopy)
    lstrSubList = lstrWordsCopy[:intCount].copy()
    return lstrSubList

def PermutationsOnProbabilities(lstrWords,dfltProbabilities,blnBackwards=False):
    '''
    Returns some permutations that are possible with a given set of words:
    - tries to order the words according to bigram probabilities.
    - choses random words if none fits by bigram probabilities.
    - sets a maximum of words at each position:
      - maximum number of permutations = max ^ wordcount.
      - example: max=2 and wordcount=20: 2^20 = 1'000'000 permutations
    - loop approach; a recursive approach may be possible
    
    When     Who What
    06.09.22 dh  Created
    '''    
    
    # init
    cintMaxWordsPerPosition = 2
    cblnDebugging = False    
    lstrCombinations = []
    lstrNewCombinations = []
    
    # define new words for each position
    for intWord in range(len(lstrWords)):
        
        # init
        if len(lstrCombinations) == 0:
            
            # find a sentence starter
            lstrCurrWords = []
            lfltProbabilities = []
            for strWord in lstrWords:
                strBigram = f"{strWord}_zzz" if blnBackwards else f"aaa_{strWord}"
                if strBigram in dfltProbabilities.keys():
                    lstrCurrWords.append(strWord)
                    lfltProbabilities.append(dfltProbabilities[strBigram])
            lstrBestWords = BestWords(lstrCurrWords,lfltProbabilities,cintMaxWordsPerPosition)
            for strWord in lstrBestWords:
                lstrCombinations.append(strWord)
            if cblnDebugging:
                print("init 1")
                print("- lstrBestWords",lstrBestWords)
                print("- lstrCombinations",lstrCombinations)
              
            # choose any word if no starter words
            if len(lstrBestWords) == 0:
                lstrAnyWords = AnyWords(lstrWords,cintMaxWordsPerPosition)
                for strWord in lstrAnyWords:
                    lstrCombinations.append(strWord)
                if cblnDebugging:
                    print("init 2")
                    print("- lstrAnyWords",lstrAnyWords)
                    print("- lstrCombinations",lstrCombinations)
            
        # words 1 to last
        else:
            lstrNewCombinations = []
            for strCombination in lstrCombinations:
                
                # determine words unused so far
                lstrUsedWords = strCombination.split(" ")
                lstrRemainingWords = lstrWords.copy()
                for strUsedWord in lstrUsedWords:
                    lstrRemainingWords.remove(strUsedWord)
                
                # prolong combinations by 1 word                
                lstrCurrWords = []
                lfltProbabilities = []

                for strRemainingWord in lstrRemainingWords:
                    strBigram = f"{strRemainingWord}_{lstrUsedWords[0]}" if blnBackwards else f"{lstrUsedWords[-1]}_{strRemainingWord}"
                    if strBigram in dfltProbabilities.keys(): 
                        lstrCurrWords.append(strRemainingWord)
                        lfltProbabilities.append(dfltProbabilities[strBigram])                        

                lstrBestWords = BestWords(lstrCurrWords,lfltProbabilities,cintMaxWordsPerPosition)
                for strBestWord in lstrBestWords:
                    if blnBackwards:
                        strNewCombination = f"{strBestWord} {strCombination}"
                    else:
                        strNewCombination = f"{strCombination} {strBestWord}"
                    lstrNewCombinations.append(strNewCombination)

                if cblnDebugging:
                    print("prolong 1")
                    print("- lstrBestWords",type(lstrBestWords),lstrBestWords)
                    print("- lstrNewCombinations",lstrNewCombinations)                    
                    
                # use random next word if none found
                if len(lstrBestWords) == 0:
                    lstrAnyWords = AnyWords(lstrRemainingWords,cintMaxWordsPerPosition)
                    for strWord in lstrAnyWords:
                        if blnBackwards:
                            lstrNewCombinations.append(f"{strWord} {strCombination}")
                        else:
                            lstrNewCombinations.append(f"{strCombination} {strWord}")

                    if cblnDebugging:
                        print("prolong 2")
                        print("- lstrAnyWords",lstrAnyWords)
                        print("- lstrNewCombinations",lstrNewCombinations)  
                
            # use these new combinations for the next loop
            lstrCombinations = lstrNewCombinations.copy()
            if cblnDebugging:
                print("Update lstrCombinations",lstrCombinations)
    
    # translate into list of tuples (for later integration
    ltstrCombinations = []
    for strCombination in lstrCombinations:
        ltstrCombinations.append(tuple(strCombination.split()))
    
    return ltstrCombinations

if False:
    dfltProbabilities = {
        "aaa_Peter":1,"Peter_und":1,"und_Paul":1,"Paul_zzz":1,
        "aaa_Paul":1,"Paul_und":1,"und_Peter":1,"Peter_zzz":0.5,
        "aaa_Peter":1,"Peter_oder":1,"oder_Paul":1,
    }
    llstrTesters = [
        ["Peter","und","Paul"],
        ["Peter","oder","Paul"],
        ["mein","Peter","oder","Paul"],
        ["mein","Peter","oder","keiner"],
        ["Peter","Peter","oder","meiner","hat","und","Paul","und","Peter"],
    ]
    
    for lstrTesters in llstrTesters:
        ltstrCombinations = PermutationsOnProbabilities(lstrTesters,dfltProbabilities,True)
        print("Number of combinations:",len(ltstrCombinations),"from",lstrTesters)
        for tstrCombination in ltstrCombinations[:10]:
            p("-",tstrCombination)
        print()           
        
def RemoveBlanksBeforePunctuation(strSource):
    '''
    Removes blanks before punctuation.
    When     Who What
    24.08.22 dh  Created
    19.09.22 dh  Single quote
    '''
    lstrPunctuations = [".","?","!",","]
    for strPunctuation in lstrPunctuations:
        strSource = strSource.replace(f" {strPunctuation}",strPunctuation)
    
    cstrNiceSingleQuote = "\u2018"
    strSource = strSource.replace(f" {cstrNiceSingleQuote}",cstrNiceSingleQuote)
    strSource = strSource.replace(f"{cstrNiceSingleQuote} ",cstrNiceSingleQuote)
    return strSource   
    
def PadList(lvarSource, intSize, varEmpty = None):
    '''
    Pads a list up to a certain length.
    Inspired by Nuno Andre (https://stackoverflow.com/questions/3438756/some-built-in-to-pad-a-list-in-python)
    When     Who What
    20.09.22 dh  Created
    '''
    return (lvarSource + intSize * [varEmpty])[:intSize]
if False:
    print(PadList([1,2,3],10))   
    
def SplitWithPunctuation(strSource,blnLowerCase=True):
    '''
    Splits a sentence into words, considering punctuation symbols as single words.
    
    When       Who What
    2022 10 21 dh  Created
    '''
    strPunctuations = "\".?!,'\u201e\u201c\xbb\xab\u201a\u2018\u203a\u2039" # UTF-8 must be translated for libraries
    lstrPunctuations = list(strPunctuations)
    if blnLowerCase:
        strSource = strSource.lower()
    for strPunctuation in lstrPunctuations:
        strSource = strSource.replace(strPunctuation,f" {strPunctuation} ")
    strSource = strSource.replace("  "," ")
    return strSource.split()
    
if False:
    strSource = "Ich liebe Dich, Monika!" 
    for strWord in SplitWithPunctuation(strSource):
        print(strWord)    
        
def NormalizedToSum1(lfltValues):
    '''
    Divides all list elements to get an element sum of 1
    When       Who What
    2022 10 26 dh  Created
    '''
    intSum = sum(lfltValues)
    for intElement,fltValue in enumerate(lfltValues):
        lfltValues[intElement] =fltValue / intSum  
        
if False:
    lintTest = [1,2,3,4]
    NormalizedToSum1(lintTest)
    print(lintTest)

def RoundWithTrailingZeroes(fltSource,intDecimals):
    '''
    Rounds floats, with trailing zeroes if necessary.
    When       Who What
    2022 10 26 dh  Created
    '''    
    return '{:.3f}'.format(round(fltSource, intDecimals))
if False:
    print(RoundWithTrailingZeroes(3.1415926,3)) # 3.142
    print(RoundWithTrailingZeroes(3,3))         # 3.000     
    
import re
def RemoveWikipediaReferences(strSource):
    '''
    Returns all square brackets and their contents. Non-greedy version.
    When       Who What
    2022 10 28 dh  Created
    '''
    return re.sub(r'\[.*?\]', "", strSource)
if False:
    strSource = "Alpha[17] Beta[18] Gamma[19]."
    print(RemoveWikipediaReferences(strSource))       