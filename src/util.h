#pragma once
#include "caffe/caffe.hpp"
#include "hack/types.h"


template <typename Dtype>caffe::Blob<Dtype> * make_blob(const struct blob_dtype_def<Dtype> & def);
template <typename Dtype>void create_blobs(std::vector<caffe::shared_ptr<caffe::Blob<Dtype>>>&blobs,const struct blob_dtype_def<Dtype> def[],int n);
