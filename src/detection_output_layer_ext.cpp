
#include "detection_output_layer_ext.h"

extern caffe::LayerParameter dummy_layer_param;

void DetectionOutputLayer_ext::setup(detection_output_param & param)
{
   _layer_param=&param;
}

DetectionOutputLayer_ext::DetectionOutputLayer_ext():caffe::DetectionOutputLayer<float>(dummy_layer_param)
{
}


void DetectionOutputLayer_ext::Reshape(const std::vector<caffe::Blob<float>*>& bottom,
      const std::vector<caffe::Blob<float>*>& top)
{
}

