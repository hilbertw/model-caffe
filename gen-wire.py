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

def print_top_wire(f,net,i):
     self=net._layer_names[i]
     top_ids=net._top_ids(i)
     n=len(top_ids)
     if n==0:
        f.write("%s.top_blob_empty.write(true);\n"%(self))
        return 
     if n==1:
       l=find_in_bottom(net,top_ids[0],i)
       if l>=0:
           bottom= net._layer_names[l]
           f.write("%s.top_blob_empty(%s.bottom_blob_empty);\n"%(self,bottom))
       else:
           f.write("%s.top_blob_empty(output_empty);\n"%(self))
     else:
       f.write("%s_top_and.clk(clk);\n"%(self))
       f.write("%s_top_and.reset(reset);\n"%(self))
       j=0
       for id in top_ids:
           l=find_in_bottom(net,id,i)
           if l>=0:
             bottom=net._layer_names[l]
             f.write("%s_top_and.in[%d](%s.bottom_blob_empty);\n"%(self,j,bottom))
           else:
             f.write("%s_top_and.in[%d](output_empty);\n"%(self,j))
              
           j=j+1

       f.write("%s.top_blob_empty(%s_top_and.out);\n"%(self,self))

def print_bottom_wire(f,net,i):
     self=net._layer_names[i]
     bottom_ids=net._bottom_ids(i)
     n=len(bottom_ids)
     if n==0:
        f.write("%s.bottom_blob_filled(input_filled);\n"%(self))
        return
     if n==1:
       l=find_in_top(net,bottom_ids[0],i)
       if l>=0:
          top= net._layer_names[l]
          f.write("%s.bottom_blob_filled(%s.top_blob_filled);\n"%(self,top))
       else:
          f.write("%s.bottom_blob_filled(input_filled);\n"%(self))
     else:
#       f.write("%s_bottom_and.clk(clk);\n"%(self))
#       f.write("%s_bottom_and.reset(reset);\n"%(self))
       j=0
       for id in bottom_ids:
           l=find_in_top(net,id,i)
           if l>=0:
              top=net._layer_names[l]
              f.write("%s_bottom_and.in[%d](%s.top_blob_filled);\n"%(self,j,top))
           else:
              f.write("%s_bottom_and.in[%d](input_filled);\n"%(self,j))
           j=j+1

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
         f.write("%s.reset(reset);\n"%(name))
         print_top_wire(f,net,i)
         print_bottom_wire(f,net,i)
         i=i+1


     s=[]
     for id in net._inputs:
         l=find_in_top(net,id,len(net._layer_names))
         if l<0:
             raise(0)
         if not l in s:
             s.append(l)
     n=len(s)
     if n<1:
          raise(0)
     if n>1:
          i=0
#          f.write("input_and.clk(clk);\n")
#          f.write("input_and.reset(reset);\n")
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
         l=find_in_top(net,id,len(net._layer_names))
         if l<0:
            raise(0)
         if not l in s:
             s.append(l)
     n=len(s)   
     if n<1:
          raise(0)

     if n>1:
          i=0
#          f.write("output_and.clk(clk);\n")
#          f.write("output_and.reset(reset);\n")
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
