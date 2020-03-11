#pragma once
#include <boost/smart_ptr/make_shared_array.hpp>
#include "systemc.h"
#include "caffe/blob.hpp"

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
    
   std::vector<caffe::Blob< float >*> top,bottom;
   void append_top(caffe::Blob< float >* b){ top.push_back(b);}
   void append_bottom(caffe::Blob< float >* b){ bottom.push_back(b);}
   void load_weight(std::vector<int>& shape,const float * data, const float * diff);
};


template<class T>
void sc_layer<T>::load_weight(std::vector<int> &shape,const float * data,const float * diff)
{
       caffe::Blob< float  > b(shape);
       b.set_cpu_data((float*)data);
       b.set_cpu_diff((float*)diff);
       //caffe::BlobProto proto;

       //b.FromProto(proto,false);

       //boost::shared_ptr<caffe::Blob< float  >> b_ptr=boost::make_shared<caffe::Blob< float  >>();
       //b_ptr->Reshape(shape);
       //b_ptr->FromProto(proto,false);  
       //caffe_layer.blobs().push_back(b_ptr);
       //caffe_layer.blobs().emplace_back();
       //caffe_layer.blobs().back()->Reshape(shape);
       //caffe_layer.blobs().back()->FromProto(proto,false);
;
}
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
