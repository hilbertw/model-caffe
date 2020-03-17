#pragma once
//#include <boost/smart_ptr/make_shared_array.hpp>
#include "systemc.h"
#include "caffe/caffe.hpp"
#include "caffe/blob.hpp"
#include "hack/types.h"

template<class T>
SC_MODULE(  sc_layer) {

    SC_CTOR( sc_layer)
    {
        SC_THREAD(run);
        //-- Sentivity list --//
        sensitive << clk.pos()<<reset;
    }

   T caffe_layer;
   sc_in<bool> clk;
   sc_in<bool> reset;
   sc_out<bool> top_blob_filled;
   sc_out<bool> bottom_blob_empty;
   sc_in<bool> top_blob_empty;
   sc_in<bool> bottom_blob_filled;

    void run();
    T & get() {return caffe_layer;}
   void forward()
   {
        caffe_layer.Forward(bottom,top);
   }
   std::vector<caffe::shared_ptr<caffe::Blob<_Dtype_>>> &blobs()
   {
       return caffe_layer.blobs();
   } 
   std::vector<caffe::Blob< float >*> top,bottom;
   void append_top(caffe::Blob< float >* b){ top.push_back(b);}
   void append_bottom(caffe::Blob< float >* b){ bottom.push_back(b);}
};


template<class T>
void sc_layer<T>::run()
{
        if(reset.read()==false)
        {
                 top_blob_filled.write(false);
                 bottom_blob_empty.write(true);
     
        }
        else if(bottom_blob_filled && top_blob_empty)
        {
                 top_blob_filled.write(false);
                 bottom_blob_empty.write(false);
                 forward();
                 top_blob_filled.write(true);
                 bottom_blob_empty.write(true);
        }
}
