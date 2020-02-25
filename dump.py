import numpy as np
import sys, os
import argparse
import caffe_pb2 as cq
from google.protobuf import text_format
def get_cpp_type(f):
        s=None
        if f.cpp_type==f.CPPTYPE_BOOL:
             s="bool"
        if f.cpp_type==f.CPPTYPE_DOUBLE:
             s="double"
        if f.cpp_type==f.CPPTYPE_ENUM:
             s="enum"
        if f.cpp_type==f.CPPTYPE_FLOAT:
             s="float"
        if f.cpp_type==f.CPPTYPE_INT32:
             s="int32"
        if f.cpp_type==f.CPPTYPE_INT64:
             s="int642"
        if f.cpp_type==f.CPPTYPE_MESSAGE:
             s="message"
        if f.cpp_type==f.CPPTYPE_STRING:
             s="string"
        if f.cpp_type==f.CPPTYPE_UINT32:
             s="uint32"
        if f.cpp_type==f.CPPTYPE_UINT64:
             s="uint64"
        if not s:
             raise(0) 
        return s

def get_label(f):
        s=None
        if f.label==f.LABEL_OPTIONAL:
               s="optional"
        if f.label==f.LABEL_REPEATED:
               s="repeated"
        if f.label==f.LABEL_REQUIRED:
               s="required"
        if not s:
             raise(0) 
        return s

def get_type(f):
        s=None
        if f.type==f.TYPE_BOOL:
              s="bool"
        if f.type==f.TYPE_BYTES:
              s="bytes"
        if f.type==f.TYPE_DOUBLE:
              s="double"
        if f.type==f.TYPE_ENUM:
              s="enum"
        if f.type==f.TYPE_FIXED32:
              s="fixed32"
        if f.type==f.TYPE_FIXED64:
              s="fixed64"
        if f.type==f.TYPE_FLOAT:
              s="float"
        if f.type==f.TYPE_GROUP:
              s="group"
        if f.type==f.TYPE_INT32:
              s="int32"
        if f.type==f.TYPE_INT64:
              s="int64"
        if f.type==f.TYPE_MESSAGE:
              s="message"
        if f.type==f.TYPE_SFIXED32:
              s="dfixed32"
        if f.type==f.TYPE_SFIXED64:
              s="sfixed64"
        if f.type==f.TYPE_SINT32:
              s="sint32"
        if f.type==f.TYPE_SINT64:
              s="sint64"
        if f.type==f.TYPE_STRING:
              s="string"
        if f.type==f.TYPE_UINT32:
              s="uint32"
        if f.type==f.TYPE_UINT64:
              s="uint64"
        if not s:
             raise(0) 
        return s
#        print("==")
#        print(f.GetOptions)
#        print(f.__class__)
#        print(f.__delattr__)
#        print(f.__doc__)
#        print(f.__format__)
#        print(f.__getattribute__)
#        print(f.__hash__)
#        print(f.__init__)

def print_fields(l):
    for f in  l.DESCRIPTOR.fields:
#        print(dir(f))
#        print(f)
#        print(f.__new__)
#        print(f.__reduce__)
#        print(f.__reduce_ex__)
#        print(f.__repr__())
#        print(f.__setattr__)
#        print(f.__sizeof__())
#        print(f.__str__())
#        print(f.__subclasshook__)
#        print(f._cdescriptor)
#        print(f._options)
#        print(f._serialized_options)
        print("camelcase_name:%s"%(f.camelcase_name))
        print(f.containing_oneof)
#        print(f.containing_type)
        print("cpp_type:%d"%(f.cpp_type))
        print("cpp type:%s"%(get_cpp_type(f)))
        if isinstance(f.default_value,int):
            print("default_value:%d"%(f.default_value))
        if isinstance(f.default_value,list):
              print("default_value:")
              for v in f.default_value:
                 print(v)

        if f.enum_type:
           print("enum_type:%s"%(f.enum_type.full_name))
        print(f.extension_scope)
#        print("file:%s"%(f.file))
        print("full_name:%s"%(f.full_name))
        print(f.has_default_value)
        print(f.has_options)
        print("id:%d"%(f.id))
        print("index:%d"%(f.index))
        print(f.is_extension)
        print("json_name:%s"%(f.json_name))
        print("label:%s"%(f.label))
        print("label:%s"%(get_label(f)))
        if f.message_type:
           print("mesage_type:%s"%( f.message_type.full_name))
        print("name:%s"%(f.name))
        print("number:%d"%(f.number))
        print("type:%d"%(f.type))
        print("type:%s"%(get_type(f)))
#        sys.exit(0)

def print_layer_param(l):

   if l.type=="Concat":
#       print_fields(l.concat_param)
       return

   if l.type=="Convolution":
       pad_w = l.convolution_param.pad_w
       pad_h = l.convolution_param.pad_h
       k_w   = l.convolution_param.kernel_w
       k_h   = l.convolution_param.kernel_h
       print_fields(l.convolution_param)
       return
   if l.type=="DetectionOutput":
       print_fields(l.detection_output_param)
       return
   if l.type=="Flatten":
#       print_fields(l.flatten_param)
       return
   if l.type=="Input":
#       print_fields(l.input_param)
       return
   if l.type=="Normalize":
       print_fields(l.norm_param)
       return
   if l.type=="Permute":
#       print_fields(l.permute_param)
       return
   if l.type=="Pooling":
#       print_fields(l.pooling_param)
       return
   if l.type=="PriorBox":
       print_fields(l.prior_box_param)
       return
   if l.type=="ReLU":
#       print_fields(l.relu_param)
       return
   if l.type=="Reshape":
#       print_fields(l.reshape_param)
       return
   if l.type=="Softmax":
#       print_fields(l.softmax_param)
       return
   if l.type=="Split":
#       print_fields(l.split_param)
       return
   raise(0)

if len(sys.argv)<2:
   sys.exit(0)

fn=sys.argv[1]
print(fn)
f = open(fn, 'r')
cq2 = cq.NetParameter()
#cq2.ParseFromString(f.read())
text_format.Parse(f.read(), cq2)

f.close()

print(dir(cq2))
#print(cq2.__attr__)
i=0
for l in cq2.layer:
   print_layer_param(l)
   print ("name %d layer: %s %s"%(i , l.name,l.type))
   i=i+1 
