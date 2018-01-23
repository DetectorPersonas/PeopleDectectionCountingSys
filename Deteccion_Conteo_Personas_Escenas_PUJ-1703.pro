#-------------------------------------------------
#
# Project created by QtCreator 2017-10-23T23:24:55
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = Deteccion_Conteo_Personas_Escenas_PUJ-1703
TEMPLATE = app


SOURCES += main.cpp\
        detector_personas.cpp

HEADERS  += detector_personas.h

FORMS    += detector_personas.ui

INCLUDEPATH +=C:\\OpenCVDirs\\opencv3\\install\\include

LIBS +=-LC:\\OpenCVDirs\\opencv3\\lib\\Debug \
    -lopencv_superres300d \
    -lopencv_surface_matching300d \
    -lopencv_text300d \
    -lopencv_tracking300d \
    -lopencv_ts300d \
    -lopencv_video300d \
    -lopencv_videoio300d \
    -lopencv_videostab300d \
    -lopencv_xfeatures2d300d \
    -lopencv_ximgproc300d \
    -lopencv_xobjdetect300d \
    -lopencv_bgsegm300d \
    -lopencv_bioinspired300d \
    -lopencv_calib3d300d \
    -lopencv_ccalib300d \
    -lopencv_core300d \
    -lopencv_datasets300d \
    -lopencv_face300d \
    -lopencv_features2d300d \
    -lopencv_flann300d \
    -lopencv_hal300d \
    -lopencv_highgui300d \
    -lopencv_imgcodecs300d \
    -lopencv_imgproc300d \
    -lopencv_latentsvm300d \
    -lopencv_line_descriptor300d \
    -lopencv_ml300d \
    -lopencv_objdetect300d \
    -lopencv_optflow300d \
    -lopencv_photo300d \
    -lopencv_reg300d \
    -lopencv_rgbd300d \
    -lopencv_saliency300d \
    -lopencv_shape300d \
    -lopencv_stitching300d
