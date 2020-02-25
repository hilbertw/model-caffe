import numpy as np
import sys, os
import argparse
import caffe_pb2 as cq
from google.protobuf import text_format

layer_param_name=dict()

def get_cpp_type(f):
        s=None
        if f.cpp_type==f.CPPTYPE_BOOL:
             s="bool"
        if f.cpp_type==f.CPPTYPE_DOUBLE:
             s="double"
        if f.cpp_type==f.CPPTYPE_ENUM:
             s="int"
        if f.cpp_type==f.CPPTYPE_FLOAT:
             s="float"
        if f.cpp_type==f.CPPTYPE_INT32:
             s="int32_t"
        if f.cpp_type==f.CPPTYPE_INT64:
             s="int64_t"
        if f.cpp_type==f.CPPTYPE_MESSAGE:
             s="message"
        if f.cpp_type==f.CPPTYPE_STRING:
             s="const char *"
        if f.cpp_type==f.CPPTYPE_UINT32:
             s="uint32_t"
        if f.cpp_type==f.CPPTYPE_UINT64:
             s="uint64_t"
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
#        print("camelcase_name:%s"%(f.camelcase_name))
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
        print("enum_type:")
        print(f.enum_type)
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
        print("mesage_type:") 
        print(f.message_type)
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

def print_optional_fields(file,message,opt_fields,indent):
    for field in opt_fields:
        d=0
        if hasattr(message,field):
             d=1
        file.write(",\n")
        i=0
        while i< indent:
             i=i+1
             file.write("  ")
        file.write("%s"%(d))

def print_repeat(file,data,field):
    d=data.__getattribute__(field.name)
    file.write("{ %d"%(len(d)))
    for v in d:
                file.write(",")
                file.write("%s"%(v))
    file.write("}")

def print_field( file,data,field,indent):
    i=0
    while i< indent:
         i=i+1
         file.write("  ")
    if field.cpp_type==field.CPPTYPE_STRING:
         file.write("\"%s\""%(data.__getattribute__(field.name)))
    else:
         #print(field.full_name)
         #print(dir(data))    
         if field.label==field.LABEL_REPEATED:
             print_repeat(file,data,field)
         else: 
             file.write("%s"%(data.__getattribute__(field.name)))
    if field.cpp_type==field.CPPTYPE_ENUM:
        file.write(" /*enum%s*/"%(field.enum_type.full_name))
    else:
        file.write(" /*%s*/"%(field.name))

def print_message( f,data,name,message_DESCRIPTOR,indent):
    i=0
    while i< indent:
         i=i+1
         file.write("  ")
    file.write("/*struct __%s*/ {\n"%(name))
    first=True
    opt_fields=[]
    for field in  message_DESCRIPTOR.fields:
           if not first:
               file.write(",\n")
           first=False
           if field.label==field.LABEL_OPTIONAL:
                opt_fields.append(field.name)
           if field.cpp_type==field.CPPTYPE_MESSAGE:
                data1=data.__getattribute__(field.name)
                print_message(file,data1,field.name,field.message_type,indent+1)
           else:
                print_field(file,data,field,indent+1)

#    print_optional_fields(file,data,opt_fields,indent+1)
    file.write("\n")
    i=0
    while i< indent:
         i=i+1
         file.write("  ")
    file.write("} /*%s*/\n"%(name))

def print_struct( file,struct_name,layer_name,message):
    file.write("struct %s %s ={\n"%(struct_name,layer_name))
    first=True
    opt_fields=[]
    for field in  message.DESCRIPTOR.fields:
           if not first:
               file.write(",\n")
           first=False
           if field.label==field.LABEL_OPTIONAL:
                opt_fields.append(field.name)
           if field.cpp_type==field.CPPTYPE_MESSAGE:
                data=message.__getattribute__(field.name)
                print_message(file,data,field.name,field.message_type,1)
           else:
                print_field( file,message,field,1)
#    print_optional_fields(file,message,opt_fields,1)
    file.write("};\n")
   
def get_field(l):

   if l.type=="Concat":
       return l.concat_param
       

   if l.type=="Convolution":
       return l.convolution_param
       
   if l.type=="DetectionOutput":
       return l.detection_output_param
       
   if l.type=="Flatten":
       return l.flatten_param
       
   if l.type=="Input":
       return l.input_param
       
   if l.type=="Normalize":
       return l.norm_param
       
   if l.type=="Permute":
       return l.permute_param
       
   if l.type=="Pooling":
       return l.pooling_param
       
   if l.type=="PriorBox":
       return l.prior_box_param
       
   if l.type=="ReLU":
       return l.relu_param
       
   if l.type=="Reshape":
       return l.reshape_param
       
   if l.type=="Softmax":
       return l.softmax_param
       
   if l.type=="Split":
       return l.split_param
       
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

with open("layers.txt","r") as f:
   for line in f:
       s_line=line.strip()
       if s_line=="":
           continue
       if s_line.startswith("#"):
           continue
       words=s_line.split()
       if len(words)<3:
            raise(0)
       param=words[2]
       fn=words[1]
       layer=words[0][:-5]
       layer_param_name[layer]=param
       #print(layer) 
       #print(words[0])

 
#for k in layer_param_name:
#   print(k)
#   print(layer_param_name[k])
#print(dir(cq2))
#print(cq2.__attr__)
i=0


with open ("gen/layer_params.cpp","w") as file:
  file.write("#include \"layer_params.h\"\n")
  file.write("#define True true\n")
  file.write("#define False false\n")
  for l in cq2.layer:
#   print ("name %d layer: %s %s"%(i , l.name,l.type))
    i=i+1
#    print(dir(l))
#    sys.exit(0)
    struct_name = layer_param_name[l.type] 
    field=get_field(l)
    print_struct(file,struct_name,l.name+"_p",field)

  file.close()


#['ByteSize', 'Clear', 'ClearExtension', 'ClearField', 'CopyFrom', 'DESCRIPTOR', 'DiscardUnknownFields', 'Extensions', 'FindInitializationErrors', 'FromString', 'HasExtension', 'HasField', 'IsInitialized', 'ListFields', 'MergeFrom', 'MergeFromString', 'ParseFromString', 'RegisterExtension', 'SerializePartialToString', 'SerializeToString', 'SetInParent', 'UnknownFields', 'WhichOneof', '_CheckCalledFromGeneratedFile', '_SetListener', '__class__', '__deepcopy__', '__delattr__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__getstate__', '__gt__', '__hash__', '__init__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__setstate__', '__sizeof__', '__slots__', '__str__', '__subclasshook__', '__unicode__', '_extensions_by_name', '_extensions_by_number', 'accuracy_param', 'annotated_data_param', 'argmax_param', 'batch_norm_param', 'bias_param', 'blobs', 'bottom', 'concat_param', 'contrastive_loss_param', 'convolution_param', 'crop_param', 'data_param', 'detection_evaluate_param', 'detection_output_param', 'dropout_param', 'dummy_data_param', 'eltwise_param', 'elu_param', 'embed_param', 'exclude', 'exp_param', 'flatten_param', 'hdf5_data_param', 'hdf5_output_param', 'hinge_loss_param', 'image_data_param', 'include', 'infogain_loss_param', 'inner_product_param', 'input_param', 'log_param', 'loss_param', 'loss_weight', 'lrn_param', 'memory_data_param', 'multibox_loss_param', 'mvn_param', 'name', 'norm_param', 'param', 'parameter_param', 'permute_param', 'phase', 'pooling_param', 'power_param', 'prelu_param', 'prior_box_param', 'propagate_down', 'python_param', 'recurrent_param', 'reduction_param', 'relu_param', 'reshape_param', 'scale_param', 'sigmoid_param', 'slice_param', 'softmax_param', 'spp_param', 'tanh_param', 'threshold_param', 'tile_param', 'top', 'transform_param', 'type', 'video_data_param', 'window_data_param']
