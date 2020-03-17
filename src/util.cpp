#include "util.h"

template <typename Dtype> Dtype * create_array( int count,Dtype init_data)
{
            Dtype *d= new Dtype[count];
            if(d)
            {
                  for(int i=0;i<count;i++) d[i]=init_data; 
            }
            return d; 
    
}

template <typename Dtype>caffe::Blob<Dtype> * make_blob(const struct blob_dtype_def<Dtype> & def)
{
      std::vector<int>shape(def.shape.count);
      for(int i=0;i<def.shape.count;i++) shape[i]=def.shape.data[i];
      
      caffe::Blob<Dtype> *b = new caffe::Blob<Dtype> (shape);
      Dtype *d;
      if(def.data_flag==1) d=def.data;
      else if(def.data_flag>1) d= create_array<Dtype>( def.count,def.data_flag==3?0:1);
      else d=NULL;
      b->set_cpu_data(d);

      if(def.diff_flag==1) d=def.data;
      else if(def.diff_flag>1) d= create_array<Dtype>( def.count,def.diff_flag==3?0:1);
      else d=NULL;
      b->set_cpu_diff(d);
 
     return b; 
}

template caffe::Blob<float> * make_blob(const blob_dtype_def<float> & def);
template caffe::Blob<double> * make_blob(const blob_dtype_def<double> & def);
template caffe::Blob<int> * make_blob(const blob_dtype_def<int> & def);

template <typename Dtype> void create_blobs(std::vector<caffe::shared_ptr<caffe::Blob<Dtype>>>&blobs,const struct blob_dtype_def<Dtype> def[],int n)
{
     for(int i=0;i<n;i++)
     {
          caffe::Blob<Dtype> * b;
          b=make_blob(def[i]);
          caffe::shared_ptr<caffe::Blob<Dtype>> ptr_b(b);
          blobs.push_back(ptr_b);
     } 
}
template void create_blobs(std::vector<caffe::shared_ptr<caffe::Blob<int>>>&blobs,const struct blob_dtype_def<int> def[],int n);
template void create_blobs(std::vector<caffe::shared_ptr<caffe::Blob<float>>>&blobs,const struct blob_dtype_def<float> def[],int n);
template void create_blobs(std::vector<caffe::shared_ptr<caffe::Blob<double>>>&blobs,const struct blob_dtype_def<double> def[],int n);
