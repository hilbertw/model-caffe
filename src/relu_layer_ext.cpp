
#include "relu_layer_ext.h"

extern caffe::LayerParameter dummy_layer_param;

void ReLULayer_ext::setup(relu_param & param)
{
   _layer_param=&param;
}

ReLULayer_ext::ReLULayer_ext():caffe::ReLULayer<float>(dummy_layer_param)
{
}


void ReLULayer_ext::Reshape(const std::vector<caffe::Blob<float>*>& bottom,
      const std::vector<caffe::Blob<float>*>& top)
{
}

