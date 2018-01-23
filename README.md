# PeopleDectectionCountingSys
People detection and counting system at indoor spaces using computer vision based strategies:
- Histogram of oriented gradients (HoG)
- SIFT extractor - Fisher Encoding (SIGFT - FV)
- Supported Vector Machine (SVM)
- Random Forest (RF)
- Convolutional Neuronal Network (CNN)
- Neuronal Network (ANN)

The following source is the result of a 6 month bachelor degree project at Pontificia Universidad Javeriana - Bogotá, about people counting systems using computer vision strategies. The final result is an executable program where user can perform the following tasks:

- Train one of the following vision algorithms using a custom database with classes "Person" and "No Person". By default, the program loads the open-source pedestrian databases of the universities of INRIA, MIT, DAIMLER and PUJ©

  - HoG / SVM or HoG / RF
  - SIFT - FV / SVM or SIFT - FV /RF
  - ANN
  - CNN

  Notes:
    - It's mandatory to firstly run training algorithms to execute people's counting system
    - The training algorithms create and place classifier trained files in the folder "ModelosEntrenados"
    - You can't retrain created models. Every time you execute the training, new files are going to replace old ones.

- Test the trained classifiers using customized image data sets.
- Test a scene to detect and count how many people are in the frame.

For more information about the folders and files of the program, remit to the following document:
https://livejaverianaedu-my.sharepoint.com/personal/n-rodriguezm_javeriana_edu_co/_layouts/15/guestaccess.aspx?folderid=0eef80b3fda604d2c86c294a7d8efb2fb&authkey=AQPr-hoA_fq_o2jDsfKLpO8&e=0337e31ab9f44a47aba77aff18a5641f

For more information about the characteristics and capabilities of the program, read the following document:
https://livejaverianaedu-my.sharepoint.com/personal/n-rodriguezm_javeriana_edu_co/_layouts/15/guestaccess.aspx?folderid=044a08e6620df40bc9327da8ad1f3f2f6&authkey=AekTTudda3hDcM9IBOl-EnA&e=67a95839a4d348f98767a9a7d2a039e3

The packages are distributed in this way:
- Installation pre-requisites: Libraries and packages to properly execute the algorithms and codes.
- Deteccion_Conteo_Personas_Escenas_PUJ-1703 : Souce code of the final program (Downloadable link)
