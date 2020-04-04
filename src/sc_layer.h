#pragma once
//#include <boost/smart_ptr/make_shared_array.hpp>
#include "systemc.h"
#include "caffe/caffe.hpp"
#include "caffe/blob.hpp"
#include "hack/types.h"
#include "hack/debug.h"

template<class T>
SC_MODULE(  sc_layer) {

    SC_CTOR( sc_layer):
       clk("clk"),
       reset("reset"),
       top_blob_filled("top_blob_filled"),
       top_blob_empty("top_blob_empty"),
       bottom_blob_filled("bottom_blob_filled"),
       bottom_blob_empty("bottom_blob_empty"),
       top_filled("top_filled"),
       bottom_empty("bottom_empty")
      
    {
        top_blob_filled(top_filled);
        bottom_blob_empty(bottom_empty);
        SC_METHOD(run);
        //-- Sentivity list --//
        sensitive << clk.pos()<<reset<<top_blob_empty<<bottom_blob_filled;
    }

   T caffe_layer;
   sc_in<bool> clk;
   sc_in<bool> reset;
   sc_out<bool> top_blob_filled;
   sc_out<bool> bottom_blob_empty;
   sc_in<bool> top_blob_empty;
   sc_in<bool> bottom_blob_filled;
   sc_signal<bool>top_filled;
   sc_signal<bool>bottom_empty;
   sc_signal<bool> forwarded;

   void run();
    T & get() {return caffe_layer;}

   void forward()
   {
BEGIN_DEBUG(name())
    // LOG(ERROR) << "Forwarding " << layer_names_[i];
    debug::print_blobs(fp,"blobs_",caffe_layer.blobs());

    debug::print_blobs(fp,"bottom_vecs_",bottom);

    debug::print_blobs(fp,"top_vecs_ before:",top);
        caffe_layer.Forward(bottom,top);
    debug::print_blobs(fp,"top_vecs_ after:",top);
END_DEBUG
   }
   std::vector<caffe::shared_ptr<caffe::Blob<_Dtype_>>> &blobs()
   {
       return caffe_layer.blobs();
   } 
   std::vector<caffe::Blob< float >*> top,bottom;
   void append_top(caffe::Blob< float >* b){ top.push_back(b);}
   void append_bottom(caffe::Blob< float >* b){ bottom.push_back(b);}
   void debug(){ 
      std::cout <<name() << ":"
                <<"top filled:" << top_blob_filled<<';'
                <<"top empty:" << top_blob_empty <<';'
                <<"bottom empty:" << bottom_blob_empty<<';'
                <<"bottom filled:" << bottom_blob_filled <<std::endl;
  }
};


template<class T>
void sc_layer<T>::run()
{
//        debug();
        if(reset.read()==false)
        {
                 forwarded.write(false);
                 top_filled.write(false);
                 bottom_empty.write(true);
     
        }
        else if (clk.posedge()) 
        {   
            if(bottom_empty.read())
            {
                 if(bottom_blob_filled && !forwarded.read())
                 {
                    bottom_empty.write(false);
                  }
            }
            else
            {
                 if(top_filled.read())
                 {
                    bottom_empty.write(true);
                 }
            } 

            if(!bottom_blob_filled.read()) forwarded.write(false);
            else if(top_filled.read()) forwarded.write(true); 
            
            
            if( top_blob_empty.read())
            {   
                if(!bottom_empty.read()&& !forwarded.read()&& !top_filled.read())
                {
                 std::cout <<name() << std::endl;
                 //getchar();
                 forward();
                 top_filled.write(true);
                 }
            }
            else 
            {
                 top_filled.write(false);
               
            }
        }
 //       debug();
}
