
#include "flatten_layer_ext.h"

extern caffe::LayerParameter dummy_layer_param;

void FlattenLayer_ext::setup(flatten_param & param)
{
   _layer_param=&param;
}

FlattenLayer_ext::FlattenLayer_ext():caffe::FlattenLayer<float>(dummy_layer_param)
{
}


void FlattenLayer_ext::Reshape(const std::vector<caffe::Blob<float>*>& bottom,
      const std::vector<caffe::Blob<float>*>& top)
{
}

