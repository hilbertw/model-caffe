
#include "input_layer_ext.h"
#include "layer_params.h"

extern caffe::LayerParameter dummy_layer_param;


InputLayer_ext::InputLayer_ext():caffe::InputLayer<float>(dummy_layer_param)
{
}


