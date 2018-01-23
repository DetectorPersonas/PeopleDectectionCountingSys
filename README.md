# PeopleDectectionCountingSys
People detection and counting system at indoor spaces using computer vision based strategies:
- Histogram of oriented gradients (HoG)
- SIFT extractor - Fisher Encoding (SIGFT - FV)
- Supported Vector Machine (SVM)
- Random Forest (RF)
- Convolutional Neuronal Network (CNN)
- Neuronal Network (ANN)

The following source is the result of a 6 month bachelor degree project at Pontificia Universidad Javeriana - Bogotá, about people counting systems using computer vision strategies. The final result is an executable program (graphical interface) where user can perform the following tasks:

- Train one of the following vision algorithms using a custom database with classes "Person" and "No Person". 

  - HoG / SVM or HoG / RF
  - SIFT - FV / SVM or SIFT - FV /RF
  - ANN
  - CNN

- Test the trained classifiers using customized image data sets: Results will include performance measures as precision, recall and F1-score of the predictions.

- Test a scene to detect and count how many people are in the frame: Results will include performance measures and will display the frame of interest with the detected people.

# General Specifications and Limitations

- By default, the program loads the open-source pedestrian databases of universities of INRIA, MIT, DAIMLER and the reservated database of Pontificia Universidad Javeriana - Bogota PUJ©.
- It's mandatory to firstly run training algorithms to execute people's counting system. These allows program to automatically create computer vision trained models.
- Although it can save many models at the same time, the program can only run one computer vision model per time.
- The model executes people detection and counting offline (User selects the scene to be analyzed)
- Program can't retrain created models. Every time a training execution is done, new models will replace old ones.

# User Manual

For more information about the folders and files of the program, remit to the file "Files and folders explanation.pdf"

For more information about program capabilities and characteristics, remit to the file "Graphic Interface capabilities and instructions.pdf"

For more information about software pre-requisites, remit to the file "Installation Pre-requisites"
