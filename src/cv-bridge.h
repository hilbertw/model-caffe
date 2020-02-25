#pragma once

namespace bridge {

int init(caffe::Blob<float> * input,caffe::Blob<float> * output);

int read_in_image();
int read_out_result();

}
