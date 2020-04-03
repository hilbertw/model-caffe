#pragma once

namespace bridge {

int init(caffe::Blob<float> * input,caffe::Blob<float> * output);

int read_in_image(const char * fn);
int read_out_result();
void set_mean(const std::string &mean_file ,const std::string & mean_value);

}
