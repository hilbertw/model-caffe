
#include "softmax_layer_ext.h"

extern caffe::LayerParameter dummy_layer_param;

void SoftmaxLayer_ext::setup(softmax_param & param)
{
   _layer_param=&param;
}

SoftmaxLayer_ext::SoftmaxLayer_ext():caffe::SoftmaxLayer<float>(dummy_layer_param)
{
}


void SoftmaxLayer_ext::Reshape(const std::vector<caffe::Blob<float>*>& bottom,
      const std::vector<caffe::Blob<float>*>& top)
{
}

