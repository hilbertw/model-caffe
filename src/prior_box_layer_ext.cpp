
#include "prior_box_layer_ext.h"

extern caffe::LayerParameter dummy_layer_param;

void PriorBoxLayer_ext::setup(prior_box_param & param)
{
   _layer_param=&param;
}

PriorBoxLayer_ext::PriorBoxLayer_ext():caffe::PriorBoxLayer<float>(dummy_layer_param)
{
}


void PriorBoxLayer_ext::Reshape(const std::vector<caffe::Blob<float>*>& bottom,
      const std::vector<caffe::Blob<float>*>& top)
{
}

