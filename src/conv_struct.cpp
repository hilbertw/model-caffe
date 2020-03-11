#include <vector>
#include <map>
#include "conv_struct.h"

template <typename Dtype> void conv_vector_dtype(std::vector<Dtype>&dest, vector_dtype_def<Dtype>& def )
{
          dest.resize(def.count);
          for(int i=0;i<def.count;i++) dest[i]=def.data[i];
}
template void conv_vector_dtype(std::vector<int>&dest, vector_dtype_def<int>& def );
template void conv_vector_dtype(std::vector<float>&dest, vector_dtype_def<float>& def );
void conv_shape(std::vector<int>&shape,struct shape_def &def)
{
        shape.resize(def.count); 
        for(int i=0;i<def.count;i++) shape[i]=def.data[i];
}
template <typename Dtype> void conv_blob_dtype(caffe::Blob<Dtype>&dest, blob_dtype_def<Dtype>& def )
{
        std::vector<int> shape;
        conv_shape(shape,def.shape);
        dest.Reshape(shape);
        dest.set_cpu_data(def.data);
        dest.set_cpu_diff(def.diff);
}
template void conv_blob_dtype(caffe::Blob<float>&dest, blob_dtype_def<float>&);
template void conv_blob_dtype(caffe::Blob<int>&dest, blob_dtype_def<int>&);
template void conv_blob_dtype(caffe::Blob<double>&dest, blob_dtype_def<double>&);

void conv_blob_int(caffe::Blob<int>&dest, blob_int_def& def )
{
        std::vector<int> shape;
        conv_shape(shape,def.shape); 
        dest.Reshape(shape);
        dest.set_cpu_data(def.data);
        dest.set_cpu_diff(def.diff);
}


void conv_blob_float(caffe::Blob<float>&dest, blob_float_def& def )
{
        std::vector<int> shape;
        conv_shape(shape,def.shape); 
        dest.Reshape(shape);
        dest.set_cpu_data(def.data);
        dest.set_cpu_diff(def.diff);
}
void conv_blob_double(caffe::Blob<double>&dest, blob_double_def& def )
{
        std::vector<int> shape;
        conv_shape(shape,def.shape); 
        dest.Reshape(shape);
        dest.set_cpu_data(def.data);
        dest.set_cpu_diff(def.diff);
}
template <typename Dtype> void conv_data_transformer(boost::shared_ptr<caffe::DataTransformer<Dtype> >&dest, data_transformer_def<Dtype>& def )
{
     conv_vector_dtype<Dtype>(dest->mean_values_,def.mean_values);
     conv_blob_dtype<Dtype>(dest->data_mean_,def.data_mean_);
}
template void conv_data_transformer<float>(boost::shared_ptr<caffe::DataTransformer<float> >&dest, data_transformer_def<float>& def );
template void conv_data_transformer<double>(boost::shared_ptr<caffe::DataTransformer<double> >&dest, data_transformer_def<double>& def );

void conv_map_int_string(std::map<int, std::string >&dest, map_int_string_def& def )
{
       
          for(int i=0;i<def.count;i++) {dest[def.data[i].first]=def.data[i].second;}
}

void conv_data_int_r( google::protobuf::RepeatedField<int>& data,struct vector_int_def &def)
{
}
void conv_data_float_r( google::protobuf::RepeatedField<float>& data,struct vector_float_def &def)
{
}
#define CONV_FIELD(x) dest.set_##x(def.x)
#define CONV_FIELD_R(x) conv_data_int_r(dest.x,def.x)
#define CONV_FIELD_R_F(x) conv_data_float_r(dest.x, def.x)

void conv_resize_param(caffe::ResizeParameter&dest, resize_param_def& def )
{
   CONV_FIELD(height);
   CONV_FIELD(width);
   CONV_FIELD(height_scale);
   CONV_FIELD(width_scale);
//   dest.set_resize_mode((caffe::ResizeParameter_Resize_mode)def.resize_mode);
//   CONV_FIELD_R(interp_mode);
//   CONV_FIELD(pad_mode);
//   CONV_FIELD_R_F(pad_value);
}
void conv_vector_int(std::vector<int>&dest, vector_int_def& def )
{
          dest.resize(def.count);
          for(int i=0;i<def.count;i++) dest[i]=def.data[i];
      
}
void conv_vector_float(std::vector<float>&dest, vector_float_def& def )
{
          dest.resize(def.count);
          for(int i=0;i<def.count;i++) dest[i]=def.data[i];
}
void conv_vector_int_ptr(std::vector<int > *&dest, vector_int_ptr_def& def )
{
          dest->resize(def.count);
          for(int i=0;i<def.count;i++) dest->data()[i]=def.data[i];
}
void conv_vector_pair_int_int(std::vector<std::pair<int, int> >&dest, vector_pair_int_int_def& def )
{
          dest.resize(def.count);
          for(int i=0;i<def.count;i++) {dest[i].first=def.data[i].first;dest[i].second=def.data[i].second;}
}
void conv_vector_string(std::vector<std::string >&dest, vector_string_def& def )
{
         
         for(int i=0;i<def.count;i++) dest.push_back(def.data[i]);  
}
