
#include "concat_layer_ext.h"

extern caffe::LayerParameter dummy_layer_param;

void ConcatLayer_ext::setup(concat_param & param)
{
   _layer_param=&param;
}

ConcatLayer_ext::ConcatLayer_ext():caffe::ConcatLayer<float>(dummy_layer_param)
{
}


void ConcatLayer_ext::Reshape(const std::vector<caffe::Blob<float>*>& bottom,
      const std::vector<caffe::Blob<float>*>& top)
{
}

