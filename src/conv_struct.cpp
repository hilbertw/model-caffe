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
template <typename Dtype> Dtype * get_data(int flag, int size)
{
        Dtype  *d=(Dtype *)malloc(size*sizeof(Dtype));
        if(flag==3) memset(d,size*sizeof(Dtype),0);
        else 
        for(int i=0;i<size;i++) d[i]=1;

        return d; 

}
template <typename Dtype> void conv_blob_dtype(caffe::Blob<Dtype>&dest, blob_dtype_def<Dtype>& def )
{
    if(def.shape.count)
    {
        std::vector<int> shape;
        conv_shape(shape,def.shape);
        dest.Reshape(shape);
        Dtype *d;
        if(def.data_flag)
        {
             d=def.data_flag==1?def.data:get_data<Dtype>(def.data_flag,def.count);

             dest.set_cpu_data(d);
        }
        if(def.diff_flag)
        {
             d=def.diff_flag==1?def.diff:get_data<Dtype>(def.diff_flag,def.count);

             dest.set_cpu_diff(d);
        }
    }
}
template void conv_blob_dtype(caffe::Blob<float>&dest, blob_dtype_def<float>&);
template void conv_blob_dtype(caffe::Blob<int>&dest, blob_dtype_def<int>&);
template void conv_blob_dtype(caffe::Blob<double>&dest, blob_dtype_def<double>&);
#if 0
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
#endif

template <typename Dtype> void conv_data_transformer(boost::shared_ptr<caffe::DataTransformer<Dtype> >&dest, data_transformer_def<Dtype>& def )
{
#if 0
     caffe::TransformationParameter param;
     caffe::DataTransformer<Dtype>*p = new caffe::DataTransformer<Dtype>(param,caffe::Phase(def.phase));
     assert(p);

     conv_vector_dtype<Dtype>(p->mean_values_,def.mean_values);
     conv_blob_dtype<Dtype>(p->data_mean_,def.data_mean_);
     boost::shared_ptr<caffe::DataTransformer<Dtype> >p_ptr(p);
     dest=p_ptr;
#endif
}
template void conv_data_transformer<float>(boost::shared_ptr<caffe::DataTransformer<float> >&dest, data_transformer_def<float>& def );
template void conv_data_transformer<double>(boost::shared_ptr<caffe::DataTransformer<double> >&dest, data_transformer_def<double>& def );

void conv_map_int_string(std::map<int, std::string >&dest, map_int_string_def& def )
{
       
          for(int i=0;i<def.count;i++) {dest[def.data[i].first]=def.data[i].second;}
}

template <typename Dtype> void conv_data_r( google::protobuf::RepeatedField<Dtype>& data, vector_dtype_def<Dtype> &def)
{
          for(int i=0;i<def.count;i++) data.add(def.data[i]);
}
#define CONV_FIELD(x) dest.set_##x(def.x)
#define CONV_FIELD_R_E(x,t) for(int i=0;i<def.x.count;i++) dest.add_##x (t(def.x.data[i]));
#define CONV_FIELD_R(x) for(int i=0;i<def.x.count;i++) dest.add_##x (def.x.data[i]);

void conv_resize_param(caffe::ResizeParameter&dest, resize_param_def& def )
{
   CONV_FIELD(height);
   CONV_FIELD(width);
   CONV_FIELD(height_scale);
   CONV_FIELD(width_scale);
   dest.set_resize_mode(caffe::ResizeParameter_Resize_mode(def.resize_mode));
   CONV_FIELD_R_E(interp_mode,caffe::ResizeParameter_Interp_mode);
   dest.set_pad_mode(caffe::ResizeParameter_Pad_mode(def.pad_mode));
   CONV_FIELD_R(pad_value);
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
void conv_vector_int_ptr(const std::vector<int > *&dest, vector_int_ptr_def& def )
{
          std::vector<int > *temp = new std::vector<int > (def.count);
          for(int i=0;i<def.count;i++) temp->data()[i]=def.data[i];
          dest=temp;
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
