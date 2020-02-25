import sys
import os

h_tmpl="""
#include "caffe/blob.hpp"
#include "caffe/layers/%s.hpp"
#include "layer_params.h"
class %s_ext: public caffe::%s<float> {
public:
%s_ext();
virtual ~%s_ext() {}


std::vector<boost::shared_ptr<caffe::Blob<float>>>&  blobs() {return blobs_;}

void setup(%s  &param);

%s _%s() const {return *_layer_param;}

%s * _layer_param;

%s
};
"""


cpp_tmpl="""
#include "%s_ext.h"

extern caffe::LayerParameter dummy_layer_param;

void %s_ext::setup(%s & param)
{
   _layer_param=&param;
}

%s_ext::%s_ext():caffe::%s<float>(dummy_layer_param)
{
}

%s
"""


methods_h="""
void Reshape(const std::vector<caffe::Blob<float>*>& bottom,
      const std::vector<caffe::Blob<float>*>& top);
"""

methods_cpp="""
void %s_ext::Reshape(const std::vector<Blob<float>*>& bottom,
      const std::vector<Blob<float>*>& top)
{
}
"""

input=[]

with open("layers.txt","r") as f:
   for line in f:
       s_line=line.strip()
       if s_line=="":
           continue
       if s_line.startswith("#"):
           continue
       words=s_line.split()
       if len(words)<3:
            raise(0)
       words.append("1")
       words.append("1")
       p=words[2]
       fn=words[1]
       c=words[0]
       input.append(words)
       m=""
       if words[3]=="1":
         m=methods_h

       with open("gen/"+fn+"_ext.h","w") as ff:
          ff.write(h_tmpl%(fn,c,c,c,c,p,p,p,p,m))
          ff.close()
       m=""
       if words[3]=="1":
         m=methods_cpp%(c)

       with open("gen/"+fn+"_ext.skel","w") as ff:
          ff.write(cpp_tmpl%(fn,c,p,c,c,c,m))
          ff.close() 


with open("gen/ext_layers.h","w") as ff:
    ff.write('#pragma once\n')

    for l in input:
       ff.write("#include \"%s_ext.h\"\n"%(l[1]))

    ff.close()
         
