#!/bin/bash
cd "$(dirname "$0")"
threshold="0.2"



if [ $# -gt 0 ];
then
threshold="$1"
fi


#while((1==1));
#do
#gdb --args \
build/ssd_detect \
./models/VGGNet/VOC0712Plus/SSD_512x512/deploy.prototxt \
./models/VGGNet/VOC0712Plus/SSD_512x512/VGG_VOC0712Plus_SSD_512x512_iter_240000.caffemodel \
--img_file aaa-tests-pedestrian-detection-systems-in-2019-models_100718929_l.jpg \
--confidence_threshold ${threshold}  

#done;

