
#include "permute_layer_ext.h"

extern caffe::LayerParameter dummy_layer_param;

void PermuteLayer_ext::setup(permute_param & param)
{
   _layer_param=&param;
}

PermuteLayer_ext::PermuteLayer_ext():caffe::PermuteLayer<float>(dummy_layer_param)
{
}


void PermuteLayer_ext::Reshape(const std::vector<caffe::Blob<float>*>& bottom,
      const std::vector<caffe::Blob<float>*>& top)
{
}

