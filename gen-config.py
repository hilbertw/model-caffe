import os
import sys

import caffe

import numpy as np
import argparse
from layer_list import layer_list

def split_type(s):
   words=s.rsplit(' ',1)
   return words[0],words[1]

def print_field(f,type,v):
	   f.write('/*%s*/\n'%(type))
	   if type=='vector<int>':
	   	   f.write("conv_vector_int(%s,conf.%s);\n"%((v,v)))
	   	   return
	   if type=='vector<float>':
	   	   f.write("conv_vector_float(%s,conf.%s);\n"%((v,v)))
	   	   return        	   	    
	   if type=='vector<string>':
	   	   f.write("conv_vector_string(%s,conf.%s);\n"%((v,v)))
	   	   return        	   	    
	   if type=='Blob<int>':
	   	   f.write("conv_blob_int(%s,conf.%s);\n"%((v,v)))  	   
	   	   return        	   	    
	   if type=='Blob<float>':
	   	   f.write("conv_blob_float(%s,conf.%s);\n"%((v,v)))  	   
	   	   return        	   	    
	   if type=='Blob<Dtype>':
	   	   f.write("conv_blob_dtype<_Dtype_>(%s,conf.%s);\n"%((v,v)))
	   	   return        	   	    
	   if type=='const(v,v)ector<int>*':
	   	   f.write("conv_vector_int_ptr(%s,conf.%s);\n"%((v,v)))
	   	   return        	   	    
	   if type=='map<int,string>' or type=='map<int, string>':
	   	   f.write("conv_map_int_string(%s,conf.%s);\n"%((v,v)))
	   	   return        	   	    
	   if type=='vector<pair<int, int> >':
	   	   f.write("conv_vector_pair_int_int(%s,conf.%s);\n"%((v,v)))
	   	   return  	   	   
	   if type=='shared_ptr<DataTransformer<Dtype> >':
	   	   f.write("conv_data_transformer(%s,conf.%s);\n"%((v,v)))
	   	   return  	   	
	   if type=='ResizeParameter':
	   	   f.write("conv_resize_param(%s,conf.%s);\n"%((v,v)))
	   	   return  	   	
	   if type=='CodeType':
	   	   f.write("%s=(caffe::CodeType)conf.%s;\n"%((v,v)))
	   	   return  	   	
	   f.write("%s=conf.%s;\n"%(v,v))    
	   f.write("%s=conf.%s;\n"%(v,v))    
     
def print_fields(f,s):
  fields=s.split(",")
  type=""
  syms=[]
#  print(fields)
  if  len(fields)==1:
     type,var= split_type(s)
     syms.append(var)     
  else:
#     print(fields[0])
     if fields[1].find(">")>=0:
     	    fields[0]=fields[0]+","+fields[1]
     	    fields.pop(1)
     type,var=split_type(fields[0])
     syms.append(var)     
     syms = syms+fields[1:]
#  print(type)
  #print(syms)  
  for s in syms:
  	 print_field(f,type.strip(),s)

if len(sys.argv)<3:
    sys.exit(0)

model=sys.argv[1]
weights=sys.argv[2]

print(model)
print(weights)


net = caffe.Net(model, caffe.TEST)
net.copy_from(weights)

#print_net(net)

with open ("gen/config_data.h","w") as f:
    f.write("""
#pragma once
#include "layer_conf.h"
""")
    i=0
    for l in net.layers:
              name=net._layer_names[i]
              i=i+1
              if l.type=='Input':
                  continue
              if l.type=='Split':
                  continue
              f.write("extern struct %sLayer_conf %s_conf;\n" %(l.type,name))

    f.close()

with open ("gen/config_layers.cpp","w") as f:
    f.write("""
#include "ext_layers.h"
#include "sc_net.h"
#include "layer_params.h"
#include "config_data.h"

void sc_net::config_layers()
{
""")

    i=0
    for l in net.layers:
              name=net._layer_names[i]
              i=i+1
              if l.type=='Input':
                  continue
              if l.type=='Split':
                  continue
              f.write("%s.get().config(%s_conf);\n" %(name,name))

    f.write("\n}")
    f.close
 

  	 
with open("gen/layer_config.cpp","w") as f:
  f.write("""
#include "layer_conf.h"
#include "ext_layers.h"
#include "conv_struct.h"
""")
  for l in layer_list:
    f.write("void %s_ext::config(struct %s_conf &conf)\n{\n"%(l[0],l[0]))
    if l[1]!="":
      lines=l[1].strip().split(";")
      for line in lines:
#         print(line)         
         line.strip(' \n')         
         line.rstrip(' \n')          
#         print(line)
         if line !="":
         	  print_fields(f,line)
    f.write("\n};\n")      
  f.close()
with open("gen/config_objs.mak","w") as f:
    f.write("CONF_OBJS := \\\n")
    i=0
    for l in net.layers:
              name=net._layer_names[i]
              i=i+1
              if l.type=='Input':
                  continue
              f.write("%s_d.o \\\n" %(name))
 

  	 
    f.close()
