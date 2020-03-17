import caffe

import numpy as np
import argparse
import os
import sys


if len(sys.argv)<3:
    sys.exit(0)

model=sys.argv[1]
weights=sys.argv[2]

print(model)
print(weights)


net = caffe.Net(model, caffe.TEST)
net.copy_from(weights)

#print_net(net)

with open("gen/net_weights.h","w") as f:
    f.write ('#include "hack/types.h"\n') 
    for name,vv in net.params.items():
        if len( vv)>0:
               f.write("extern const struct blob_dtype_def<_Dtype_> %s_blobs[];\n"%(name))
    f.close()

with open("gen/net_weights.cpp","w") as f:
    f.write("""
#include "net_weights.h"
#include "sc_net.h"
#include "util.h"

void sc_net::load_weights()
{
""" )
    
    for name,v in net.params.items():
       if len(v)>0:
           f.write('::create_blobs(%s.blobs(),%s_blobs,%d);\n'%(name,name,len(v)))
            

    f.write("\n}")

with open("gen/w_objs.mak","w") as f:
   f.write("WEIGHT_OBJS=\\\n")
   for name,v in net.params.items():
       if len(v)>0:
          f.write("%s_w.o \\\n"%(name))
   f.close()

#  print('convert layer: %s'%( name))

#  save_blob(output_path + '/' + str(name),layer)
#  num = 0
#  for p in net.params[name]:
#      print(dir(p))
#      np.save(output_path + '/' + str(name) + '_' + str(num), p.data)
#    save_param(output_path + '/' + str(name) , p.data)
#    f.write("""
##pragma once
#struct net_param_blob 
#{
#    int num,channels,width,height;
#    const int data_dim_len;
#    const int * data_dim;
#    const float * data;
#    const int diff_dim_len;
#    const int * diff_dim;
#    const float *diff;
#};
#""") 
