import caffe

import numpy as np
import argparse
import os
import sys


def find_in_bottom(net,id,start):
    i=start+1
    while i<len( net.layers):
        ids=net._bottom_ids(i)
        if id in ids:
           return i
        i=i+1
    return -1

def find_in_top(net,id,start):
    i=start-1
    while i>=0:
        ids=net._top_ids(i)
        if id in ids:
           return i
        i=i-1
    return -1
   
def find_top_layer(net,i):
    s=[]
    for id in net._top_ids(i):
        l=find_in_bottom(net,id,i)
        s.append(l)
    return s

def find_bottom_layer(net,i):
    s=[]
    for id in net._bottom_ids(i):
        l=find_in_top(net,id,i)
        s.append(l)
    return s

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
         top_layers=find_top_layer(net,i)
         n=len(top_layers)
         if n>1:
              f.write("sc_and<%d> %s_top_and{\"%s_top_and\"};\n"%(n,name,name))
         bottom_layers=find_bottom_layer(net,i)
         n=len(bottom_layers)
         if n>1:
              f.write("sc_and<%d> %s_bottom_and{\"%s_bottom_and\"};\n"%(n,name,name))
         i=i+1
     s=[]
     for id in net._inputs:
         l=find_in_top(net,id,len(net._layer_names))
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
         l=find_in_top(net,id,len(net._layer_names))
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

with open("gen/net_debug.cpp","w") as f:
     f.write("""
#include "sc_net.h"
void sc_net::debug()
{
""")
     i=0
     for l in net.layers:
         name=net._layer_names[i]
         f.write("%s.debug();\n"%(name))
         i=i+1

     i=0
     for l in net.layers:
         name=net._layer_names[i]
         top_layers=find_top_layer(net,i)
         n=len(top_layers)
         if n>1:
              f.write("%s_top_and.debug();\n"%(name))
         bottom_layers=find_bottom_layer(net,i)
         n=len(bottom_layers)
         if n>1:
              f.write("%s_bottom_and.debug();\n"%(name))
         i=i+1
     s=[]
     for id in net._inputs:
         l=find_in_top(net,id,len(net._layer_names))
         if l<0:
             raise(0)
         if not l in s:
             s.append(l)
     n=len(s)   
     if n <1:
         raise(0)    
     if n>1:
              f.write("input_and.debug();\n")
    
     s=[]
     for id in net._outputs:
         l=find_in_top(net,id,len(net._layer_names))
         if l<0:
            print(id)
            raise(0)
         if not l in s:
             s.append(l)
     n=len(s)   
     if n <1:
         raise(0)    
     if n>1:
         f.write("output_and.debug();\n")
     f.write("\n}\n")
     f.close() 
