
#include "split_layer_ext.h"

extern caffe::LayerParameter dummy_layer_param;


SplitLayer_ext::SplitLayer_ext():caffe::SplitLayer<float>(dummy_layer_param)
{
}


