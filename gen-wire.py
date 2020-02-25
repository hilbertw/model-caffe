import caffe

import numpy as np
import argparse
import os
import sys



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

def print_top_wire(f,net,i):
     self=net._layer_names[i]
     top_ids=net._top_ids(i)
     n=len(top_ids)
     if n==0:
        return 
     if n==1:
       l=find_in_bottom(net,top_ids[0])
       if l>=0:
           bottom= net._layer_names[l]
           f.write("%s.top_blob_empty(%s.bottom_blob_empty);\n"%(self,bottom))
       else:
           f.write("%s.top_blob_empty(output_empty);\n"%(self))
     else:
       i=0
       for id in top_ids:
           l=find_in_bottom(net,id)
           if l>=0:
             bottom=net._layer_names[l]
             f.write("%s_top_and.in[%d](%s.bottom_blob_empty);\n"%(self,i,bottom))
           else:
             f.write("%s_top_and.in[%d](output_empty);\n"%(self,i))
              
           i=i+1

       f.write("%s.top_blob_empty(%s_top_and.out);\n"%(self,self))

def print_bottom_wire(f,net,i):
     self=net._layer_names[i]
     bottom_ids=net._bottom_ids(i)
     n=len(bottom_ids)
     if n==0:
        return
     if n==1:
       l=find_in_top(net,bottom_ids[0])
       if l>=0:
          top= net._layer_names[bottom_ids[0]]
          f.write("%s.bottom_blob_filled(%s.top_blob_filled);\n"%(self,top))
       else:
          f.write("%s.bottom_blob_filled(input_filled);\n"%(self))
     else:
       i=0
       for id in bottom_ids:
           l=find_in_top(net,id)
           if l>=0:
              top=net._layer_names[l]
              f.write("%s_bottom_and.in[%d](%s.top_blob_filled);\n"%(self,i,top))
           else:
              f.write("%s_bottom_and.in[%d](input_filled);\n"%(self,i,top))
           i=i+1

       f.write("%s.bottom_blob_filled(%s_bottom_and.out);\n"%(self,self))

if len(sys.argv)<3:
    sys.exit(0)

model=sys.argv[1]
weights=sys.argv[2]

print(model)
print(weights)


net = caffe.Net(model, caffe.TEST)
net.copy_from(weights)

#print_net(net)

with open("gen/net_wire.cpp","w") as f:
     f.write(
"""
#include "sc_net.h"
void sc_net::setup_wires()
{
"""
     )

     i=0
     for l in net.layers:
         name=net._layer_names[i]
         f.write("%s.clk(clk);\n"%(name))
         print_top_wire(f,net,i)
         print_bottom_wire(f,net,i)
         i=i+1


     s=[]
     for id in net._inputs:
         l=find_in_bottom(net,id)
         if l<0:
             raise(0)
         if not l in s:
             s.append(l)
     n=len(s)
     if n<1:
          raise(0)
     if n>1:
          i=0
          for ss in s:
              name=net._layer_names[ss]
              f.write("input_and.in[%d](%s.bottom_blob_empty);\n"%(i,name))
              i=i+1
          f.write("input_empty(input_and.out);\n")
     else:
          name=net._layer_names[s[0]]
          f.write("input_empty(%s.bottom_blob_empty);\n"%(name))

          
     s=[]
     for id in net._outputs:
         l=find_in_top(net,id)
         if l<0:
            raise(0)
         if not l in s:
             s.append(l)
     n=len(s)   
     if n<1:
          raise(0)

     if n>1:
          i=0
          for ss in s:
              name=net._layer_names[ss]
              f.write("output_and.in[%d](%s.top_blob_filled);\n"%(i,name))
              i=i+1
          f.write("output_filled(output_and.out);\n")
     else:
          name=net._layer_names[s[0]]
          f.write("output_filled(%s.top_blob_filled);\n"%(name))

     f.write("\n}")
     f.close()
