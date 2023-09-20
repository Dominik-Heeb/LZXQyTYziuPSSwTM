# MonReader

Apziva project #4<br>
2023 07 29ff

__Introduction__
* MonReader is a __mobile document digitization experience__ by AI LABS for the blind, for researchers and for anyone else who needs __fully automated, fast and high-quality scanning of documents__ in bulk.
* When browsing through a book, the mobile phone app determines when the __browsing process is idle__ for a short time. In this moment of pausing, the text is captured using __optical character recognition__ (OCR). At the end of the browsing process, the __contents of the book__ are __summarized__ for the user.
* The GitHub repository shown here focusses on the __sub-process for detecting the idle moments__ in the browsing process.
  
__Data Description__
* The mobile phone camera furnished __color images with a width of 1080 and a height of 1920 pixels__, i.e. the images are in portrait format.
* For both training and testing __2989 images__ were available.
  
__Methodology__ 
* __Inception__ and __MobileNet__ were used as convolutional neural networks (__CNN__) for the initial image analysis (__transfer learning model__ or __embedding__).
* To get the images ready for processing by Inception (i.e. images of 299x299 pixels) or by MobileNet (224x224 pixels) two different resizing methods were checked: __cropping and shrinking__.
* The __high-level image features__ (HLF) were then submitted to a principal component analysis (__PCA__) to reduce the feature space.
* Finally, the pre-processed data was trained using the following models: __Ridge Regression (L2), Random Forest, AdaBoost, XGBoost, and Support Vector Machine (SVM)__, exploring a wide range of __hyperparameters__ with each model candidate.
  
__Conclusions__
* __Inception__ was identified as the transfer-learning model with the highest F1 score. However, Inception requires 92 MB of memory, which currently exceeds the typical size for mobile applications. For this __practical reason__, Inception had to be discarded. __MobileNet__, however, performs almost as well as Inception, with an F1 score of 98.6% rather than 99.1%.
* The __best model__ on MobileNet high-level features showed to be a __SVM__ using C=5 and gamma=0.01 on __cropped images__ and applying a PCA using __50 principal components__.
* A __productive version__ of this SVM is accessible at https://www.trigonella.ch/VideoAnalysis.php. This productive version was created using __Gradio__.
