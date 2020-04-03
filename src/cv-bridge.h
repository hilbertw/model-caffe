#pragma once

namespace bridge {

int init(caffe::Blob<float> * input,caffe::Blob<float> * output);

int read_in_image(const char * fn);
int read_out_result();

}
