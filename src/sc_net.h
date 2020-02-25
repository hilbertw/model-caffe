#include "systemc.h"
#include "caffe/blob.hpp"

#include "sc_and.h"
#include "sc_layer.h"
#include "ext_layers.h"

SC_MODULE(sc_net)
{

  SC_CTOR(sc_net)
  {
  }
  sc_in<bool> clk;
  sc_in<bool> input_filled;
  sc_in<bool> output_empty;
  sc_out<bool> output_filled;
  sc_out<bool> input_empty;


  void init();
  void load_weights();
  void  create_blobs();
  void  setup_blobs();
  void   setup_layers();
  void   setup_wires();

  std::vector<caffe::Blob< float >*> blobs;
  std::vector<caffe::Blob< float >*> input_blobs;
  std::vector<caffe::Blob< float >*> output_blobs;
  
#include "submodules.h"

};
