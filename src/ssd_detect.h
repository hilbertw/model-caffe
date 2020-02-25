#pragma once
#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include <iomanip>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>


using namespace std;

class Detector {
 public:
  Detector( caffe::Blob<float>*in,
            caffe::Blob<float>*out);

  std::vector<vector<float> > Detect_out();
  void Detect_in(const cv::Mat& img);

 
  void SetMean(const string& mean_file, const string& mean_value);
private:

  void WrapInputLayer(std::vector<cv::Mat>* input_channels);

  void Preprocess(const cv::Mat& img,
                  std::vector<cv::Mat>* input_channels);

 private:
  caffe::Blob<float>* input_layer;
  caffe::Blob<float>* result_blob;
  cv::Size input_geometry_;
  unsigned int num_channels_;
  cv::Mat mean_;
};

