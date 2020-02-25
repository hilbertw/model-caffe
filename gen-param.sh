#!/bin/bash


python gen-param-h.py adas_caffe_depth/models/VGGNet/VOC0712/SSD_300x300/deploy.prototxt
python gen-param-cpp.py adas_caffe_depth/models/VGGNet/VOC0712/SSD_300x300/deploy.prototxt
