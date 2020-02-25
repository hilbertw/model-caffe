
#include "reshape_layer_ext.h"

extern caffe::LayerParameter dummy_layer_param;

void ReshapeLayer_ext::setup(reshape_param & param)
{
   _layer_param=&param;
}

ReshapeLayer_ext::ReshapeLayer_ext():caffe::ReshapeLayer<float>(dummy_layer_param)
{
}


void ReshapeLayer_ext::Reshape(const std::vector<caffe::Blob<float>*>& bottom,
      const std::vector<caffe::Blob<float>*>& top)
{
}

