#!/bin/bash


python3 gen-setup-cpp.py adas_caffe_depth/models/VGGNet/VOC0712/SSD_300x300/deploy.prototxt  adas_caffe_depth/models/VGGNet/VOC0712/SSD_300x300/VGG_VOC0712_SSD_300x300_iter_120000.caffemodel
#python3 gen-weight-cpp.py adas_caffe_depth/models/VGGNet/VOC0712/SSD_300x300/deploy.prototxt  adas_caffe_depth/models/VGGNet/VOC0712/SSD_300x300/VGG_VOC0712_SSD_300x300_iter_120000.caffemodel
