TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += \
        data.cpp \
        main.cpp \
        net.cpp \
        neuron.cpp

HEADERS += \
    data.h \
    net.h \
    neuron.h \
    types.h
