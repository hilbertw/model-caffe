#pragma once
#include "systemc.h"


template<class T>
SC_MODULE(  sc_layer) {
    SC_CTOR (sc_layer) {
        // Nothing in constructor
    }
   T caffe_layer;

};

