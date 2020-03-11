
#pragma once
#include "hack/types.h"
template <typename Dtype>void conv_blob_dtype(caffe::Blob<Dtype>& dest,blob_dtype_def<Dtype>& def);
template <typename Dtype>void conv_data_transformer(caffe::DataTransformer<Dtype>& dest,data_transformer_def<Dtype> & def);
void conv_map_int_string(std::map<int,std::string> & dest,map_int_string_def & def);
void conv_resize_param(caffe::ResizeParameter & dest,resize_param_def & def);
template <typename Dtype>void conv_vector_dtype(std::vector<Dtype>& dest,vector_dtype_def<Dtype>& def);
void conv_vector_pair_int_int(std::vector<std::pair<int,int>> & dest,vector_pair_int_int_def & def);
void conv_vector_string(std::vector<std::string> & dest,vector_string_def & def);
void conv_vector_int(std::vector<int> & dest,vector_int_def & def);
void conv_vector_float(std::vector<float> & dest,vector_float_def & def);
void conv_blob_float(caffe::Blob<float> & dest,blob_float_def & def);
void conv_blob_int(caffe::Blob<int> & dest,blob_int_def & def);
template <typename Dtype>void conv_data_transformer(boost::shared_ptr<caffe::DataTransformer<Dtype> >&, data_transformer_def<Dtype>&);
void conv_vector_int_ptr(const  std::vector<int>* &dest,vector_int_ptr_def&);
