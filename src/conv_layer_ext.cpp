
#include "conv_layer_ext.h"

extern caffe::LayerParameter dummy_layer_param;

void ConvolutionLayer_ext::setup(conv_param & param)
{
   _layer_param=&param;
}

ConvolutionLayer_ext::ConvolutionLayer_ext():caffe::ConvolutionLayer<float>(dummy_layer_param)
{
}


void ConvolutionLayer_ext::Reshape(const std::vector<caffe::Blob<float>*>& bottom,
      const std::vector<caffe::Blob<float>*>& top)
{
}

