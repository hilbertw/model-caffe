import caffe

import numpy as np
import argparse
import os
import sys


   

def print_vec(t,v):
    print(t)
    for e in v:
       print(e)


def print_vvec(t,v):
    print(t)
    for vv in v:
      for e in vv:
        print(e)
      print("===")

def print_layer(l):
    print(dir(l))
    #print(l.type)
    #print_layer_param(l)
    print("blobs:%d\n"%(len(l.blobs)))
    #print_vec(l.__dict__)
    sys.exit(0)

def print_blob(l):
    #print(dir(l))
    #print_layer_param(l)
    #print("blobs:%d\n"%(len(l.blobs)))
    #print_vec(l.__dict__)
    print("%s,%s,%s,%s,%s"%(l.channels,l.count,l.height,l.num,l.width))
    for d in l.shape:
        print(d)
    print(len(l.data))
    print(len(l.diff))

def print_net(n):
    
    #print_vec("backward()backward()",n.backward())
    #print_vec("_batch",n._batch())
    #print_vec("_blob_loss_weights",n._blob_loss_weights)
   
    print_vec("_blob_names:",n._blob_names)
    for b in n._blobs:
        print_blob(b)
    i=0
    for l in n.layers:
      print_vec("_bottom_ids():",n._bottom_ids(i))
      print_vec("_top_ids():",n._top_ids(i))
      i=i+1
#    print(dir(n._forward))
#    print(dir(n._inputs))
#    print_vec("_layer_names:",n._layer_names)
#    print(dir(n._outputs))

    print_vec("outputs:",n.outputs)
    print_vec("inputs:",n.inputs)
#    for l in n.layers:
#       print_layer(l)
#    print(dir(n.params))
#    print(dir(n.params.__class__))
#    for k in n.params.keys():
#         print (k)
#    print(dir(n.blob_loss_weights))
#    print(dir(n.blobs))
    print_vec("bottom_names:",n.bottom_names)
    print_vec("top_names:",n.top_names)
#clear_param_diffs
#copy_from
#backward
#forward
#forward_all
#forward_backward_all
#load_hdf5
#reshape
#save
#save_hdf5
#set_input_arrays
#_set_input_arrays
#share_with



if len(sys.argv)<3:
    sys.exit(0)

model=sys.argv[1]
weights=sys.argv[2]

print(model)
print(weights)


net = caffe.Net(model, caffe.TEST)
net.copy_from(weights)

#print(len(net._blob_names))
#print(len(net.top_names))
#print(len(net.bottom_names))
#print_net(net)
#sys.exit(0)

with open("gen/net_blobs.cpp","w") as f:
    f.write("""
#include "sc_net.h"
void sc_net::create_blobs()
{
blobs.resize(%d);
"""
    %(len(net._blobs)))
    i=0
    for b in net._blobs:
      f.write("/*%s*/\n"%(net._blob_names[i]))
      f.write("blobs[%d]=new caffe::Blob<float>(%d,%d,%d,%d);\n"%(i,b.num,b.channels,b.height,b.width))
      i=i+1

    f.write("\n}")
    f.write("void sc_net::setup_blobs()\n{\n")
    i=0
    for l in net.layers:
       name=net._layer_names[i]
       for id in net._top_ids(i):
         f.write("/*%s*/\n"%(net._blob_names[id]))
         f.write("%s.append_top(blobs[%d]);\n"%(name,id))
       for id in net._bottom_ids(i):
         f.write("/*%s*/\n"%(net._blob_names[id]))
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
