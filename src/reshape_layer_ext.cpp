#include <assert.h>
#include "reshape_layer_ext.h"

extern caffe::LayerParameter dummy_layer_param;

void ReshapeLayer_ext::setup(reshape_param & param)
{
   _layer_param=&param;
}

ReshapeLayer_ext::ReshapeLayer_ext():caffe::ReshapeLayer<float>(dummy_layer_param)
{
}


void ReshapeLayer_ext::Reshape(const std::vector<caffe::Blob<float>*>& bottom,
      const std::vector<caffe::Blob<float>*>& top)
{
}
void ReshapeLayer_ext::Forward_cpu(const std::vector<caffe::Blob<float>*>& bottom,
      const std::vector<caffe::Blob<float>*>& top)
{
        for(int i=0;i<bottom.size();i++)
        {
            int count= bottom[i]->count();
            assert(count==top[i]->count());
            caffe::caffe_copy(count, bottom[i]->cpu_data(), top[i]->mutable_cpu_data());
        }
}

