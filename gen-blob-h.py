import caffe

import numpy as np
import argparse
import os
import sys


def print_layer_param(l):

   if l.type=="Concat":
       print(dir(l.concat_param))
       return

   if l.type=="Convolution":
       pad_w = l.convolution_param.pad_w
       pad_h = l.convolution_param.pad_h
       k_w   = l.convolution_param.kernel_w
       k_h   = l.convolution_param.kernel_h
       print(dir(l.convolution_param))
       return
   if l.type=="DetectionOutput":
       return
   if l.type=="Flatten":
       return
   if l.type=="Input":
       return
   if l.type=="Normalize":
       return
   if l.type=="Permute":
       return
   if l.type=="Pooling":
       return
   if l.type=="PriorBox":
       return
   if l.type=="ReLU":
       return
   if l.type=="Reshape":
       return
   if l.type=="Softmax":
       return
   if l.type=="Split":
       return
   raise(0)
   

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



def print_array(f,a,ident):

    if isinstance(a,np.float32):
               f.write("%f"%(a))
    else:
       i=0
       f.write("\n")
       while i<ident:
          f.write(" ")
          i=i+1
       f.write("{\n")
       first=True
       i=-2
       while i<ident:
          f.write(" ")
          i=i+1
       for e in a:
               if not first:
                  f.write(",")
               first=False
               print_array(f,e,ident+1)
       i=0
       f.write("\n")
       while i<ident:
          f.write(" ")
          i=i+1
       f.write("}")

def save_blob(n,d):
    with open(n+".h","w") as f:
      f.write("static int blob_size=%d;\n"%(len(d)))
      i=0 
      for e in d:
#        f.write("%s\n"%(dir(e)))
        f.write("const static int channels_%d=%d;\n"%(i,e. channels)) 
        f.write("const static int count_%d=%d;\n"%(i,e.  count))
        f.write("const static int data_%d=%d;\n"%(i,len(e.  data))) 
        f.write("const static int diff_%d=%d;\n"%(i,len(e.  diff)))
        f.write("const static int height_%d=%d;\n"%(i,e.  height)) 
        f.write("const static int width_%d=%d;\n"%(i,e.  width))
        f.write("const static int num_%d=%d;\n"%(i,e.num))
        f.write("static int shape_%d[]={"%(i))
        first=True
        for s in e.shape:
           if not first:
                f.write(",")
           first=False
           f.write("%d"%(s))
        i=i+1
        f.write("};\n")
      f.write( "Blob blobs[]={\n")
      first=True
      for e in d:
           if not first:
                f.write(",")
           first=False
           f.write("Blob(%d,%d,%d,%d)"%(e.num,e.channels,e.height,e.width))
      f.write("};\n")
 
      f.close()

def save_param(n,d):
    with open(n+".h","a") as f:
       f.write('static int param_shape[]={')
       first=True
       for i in d.shape:
         if not first:
            f.write(",")
         first=False
         f.write("%d"%(i))
       f.write("};\n")
       f.write('static float parameter')
       for i in d.shape:
         f.write("[%d]"%(i))
       f.write("=\n")
       print_array(f,d,0) 

       f.close()

def print_blob_data(f,i,label,data):
    f.write("const static int %s_dim_%d[]={ "%(label,i))
    first=True
    for d in data.shape:
       if not first:
           f.write(",")
       first=False
       f.write("%s"%(d))
    f.write(" };\n")
#    arr= np.reshape(data,-1)
#    if np.count_nonzero(data)<1:
#       f.write("const static float %s_%d[%d];\n"%(label,i,len(arr)))
#    else:
#       first=True
#       j=1
#       f.write("const static float %s_%d[]="%(label,i))
#       for d in arr:
#          if not first:
#              f.write(",")
#          if j==100:
#              f.write("\n")
#              j=1
#          first=False
#          f.write("%s"%(d)) 
#          j=j+1
#       f.write("\n};\n")

def print_blob_def(f,i,b):
    f.write("const  struct net_blob net_blob_%d={\n"%(i))
    f.write("%s,\n"%(b.num))
    f.write("%s,\n"%(b.channels))
    f.write("%s,\n"%(b.width))
    f.write("%s,\n"%(b.height))
    f.write("%s,\n"%(len(b.data.shape)))
    f.write("data_dim_%d,\n"%(i))
    f.write("%s,\n"%(len(b.diff.shape)))
    f.write("diff_dim_%d\n"%(i))
    f.write("\n};\n")

#def print_layer_params(input_file):
#    with open(input_file, 'r') as fp:
#        net_file = caffe_pb2.NetParameter()
#        text_format.Parse(fp.read(), net_file)
#        print(dir(net_file))
      #for l in net_file.layer:
      #      print(dir(l))
if len(sys.argv)<3:
    sys.exit(0)

model=sys.argv[1]
weights=sys.argv[2]

print(model)
print(weights)


net = caffe.Net(model, caffe.TEST)
net.copy_from(weights)

#print_net(net)

with open("gen/net_blobs.h","w") as f:
    f.write("""
struct net_blob 
{
    int num,channels,width,height;
    const int data_dim_len;
    const int * data_dim;
    //const float * data;
    const int diff_dim_len;
    const int * diff_dim;
    //const float *diff;
};
""") 
    i=0
    for b in net._blobs:
      print_blob_data(f,i,"data",b.data)
      print_blob_data(f,i,"diff",b.diff)
      print_blob_def(f,i,b)
      i=i+1

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
