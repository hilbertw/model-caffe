
#include "pooling_layer_ext.h"

extern caffe::LayerParameter dummy_layer_param;

void PoolingLayer_ext::setup(pooling_param & param)
{
   _layer_param=&param;
}

PoolingLayer_ext::PoolingLayer_ext():caffe::PoolingLayer<float>(dummy_layer_param)
{
}


void PoolingLayer_ext::Reshape(const std::vector<caffe::Blob<float>*>& bottom,
      const std::vector<caffe::Blob<float>*>& top)
{
}

