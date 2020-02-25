import caffe

import numpy as np
import argparse
import os
import sys


def find_in_bottom(net,id):
    i=0
    for l in net.layers:
        ids=net._bottom_ids(i)
        if id in ids:
           return i
        i=i+1
    return -1

def find_in_top(net,id):
    i=0
    for l in net.layers:
        ids=net._top_ids(i)
        if id in ids:
           return i
        i=i+1
    return -1
   

def print_vec(t,v):
    print(t)
    for e in v:
       print(e)

def print_layer(l):
    #print(l.type)
    #print_layer_param(l)
    print("blobs:%d\n"%(len(l.blobs)))
    #print_vec(l.__dict__)


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
    
#    print_vec("_bottom_ids():",n._bottom_ids())
#    print_vec("_top_ids():",n._top_ids())
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

#print_net(net)

with open("gen/submodules.h","w") as f:
     i=0
     for l in net.layers:
         name=net._layer_names[i]
         f.write("sc_layer<%sLayer_ext> %s{\"%s\"};\n"%(l.type,name,name))
         i=i+1

     i=0
     for l in net.layers:
         name=net._layer_names[i]
         top_ids=net._top_ids(i)
         n=len(top_ids)
         if n>1:
              f.write("sc_and<%d> %s_top_and{\"%s_top_and\"};\n"%(n,name,name))
         bottom_ids=net._bottom_ids(i)
         n=len(bottom_ids)
         if n>1:
              f.write("sc_and<%d> %s_bottom_and{\"%s_bottom_and\"};\n"%(n,name,name))
         i=i+1
     s=[]
     for id in net._inputs:
         l=find_in_bottom(net,id)
         if l<0:
             raise(0)
         if not l in s:
             s.append(l)
     n=len(s)   
     if n <1:
         raise(0)    
     if n>1:
              f.write("sc_and<%d> input_and{\"input_and\"};\n"%(n))
    
     s=[]
     for id in net._outputs:
         l=find_in_top(net,id)
         if l<0:
            print(id)
            raise(0)
         if not l in s:
             s.append(l)
     n=len(s)   
     if n <1:
         raise(0)    
     if n>1:
              f.write("sc_and<%d> output_and{\"output_and\"};\n"%(n))
     f.close() 
