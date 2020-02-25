
#include "normalize_layer_ext.h"

extern caffe::LayerParameter dummy_layer_param;

void NormalizeLayer_ext::setup(norm_param & param)
{
   _layer_param=&param;
}

NormalizeLayer_ext::NormalizeLayer_ext():caffe::NormalizeLayer<float>(dummy_layer_param)
{
}


void NormalizeLayer_ext::Reshape(const std::vector<caffe::Blob<float>*>& bottom,
      const std::vector<caffe::Blob<float>*>& top)
{
}

