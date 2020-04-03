import caffe

import numpy as np
import argparse
import os
import sys
from net_wrapper import  net_wrapper 

   


if len(sys.argv)<3:
    sys.exit(0)

model=sys.argv[1]
weights=sys.argv[2]

print(model)
print(weights)


net = caffe.Net(model, caffe.TEST)
net.copy_from(weights)

net_=net_wrapper(net)

with open("gen/net_blobs.cpp","w") as f:
    f.write("""
#include "sc_net.h"
void sc_net::create_blobs()
{
blobs.resize(%d);
"""
    %(len(net_._blobs)))
    i=0
    for b in net_._blobs:
      f.write("/*%s*/\n"%(net_._blob_names[i]))
      f.write("blobs[%d]=new caffe::Blob<float>(%d,%d,%d,%d);\n"%(i,b.num,b.channels,b.height,b.width))
      i=i+1

    f.write("\n}")
    f.write("void sc_net::setup_blobs()\n{\n")
    i=0
    for l in net.layers:
       name=net._layer_names[i]
       for id in net_._top_ids(i):
         f.write("/*%s*/\n"%(net_._blob_names[id]))
         f.write("%s.append_top(blobs[%d]);\n"%(name,id))
       for id in net_._bottom_ids(i):
         f.write("/*%s*/\n"%(net_._blob_names[id]))
         f.write("%s.append_bottom(blobs[%d]);\n"%(name,id))

       i=i+1
    for id in net._inputs:
         f.write("/*%s*/\n"%(net._blob_names[id]))
         f.write("input_blobs.push_back(blobs[%d]);\n"%(id))

    for id in net._outputs:
         f.write("/*%s*/\n"%(net._blob_names[id]))
         f.write("output_blobs.push_back(blobs[%d]);\n"%(id))

    f.write("\n}")
    f.close

#for item in net.params.items():
#  name, layer = item
#  print('convert layer: %s'%( name))

#  save_blob(output_path + '/' + str(name),layer)
#  num = 0
#  for p in net.params[name]:
#      print(dir(p))
#      np.save(output_path + '/' + str(name) + '_' + str(num), p.data)
#    save_param(output_path + '/' + str(name) , p.data)
