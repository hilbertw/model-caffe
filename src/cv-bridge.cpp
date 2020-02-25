#include "ssd_detect.h"
#include "cv-bridge.h"

namespace bridge{


Detector * detector; 
int init(caffe::Blob<float>* input_blob,caffe::Blob<float>* output_blob)
{

    detector=new  Detector(input_blob,output_blob);    
    return detector==NULL?-1:0;
}

int read_in_image()
{
   return 0;
}

int read_out_result()
{
   return 0;
}

}
