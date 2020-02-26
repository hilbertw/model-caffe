import os
import sys
from layer_list import layer_list

def split_type(s):
   words=s.rsplit(' ',1)
   return words[0],words[1]

def gen_dump(s):
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
  	yield [type,s]


caffe_path="caffe"
if len(sys.argv)>1:
    caffe_path=argv[1]

output_path=caffe_path+"/src/hack/"
if not os.path.exists(output_path):
    os.makedirs(output_path)

with open(output_path+"layer_print.cpp","w") as f:
  f.write("#include \"hack/layer_conf.h\"\n")
  f.write("#include \"hack/print_data.h\"\n")
  s=[]     
  for l in layer_list:
        if l[1]!="":
          lines=l[1].strip().split(";")
          for line in lines:
    #         print(line)         
             line.strip(' \n')         
             line.rstrip(' \n')          
    #         print(line)
             if line !="":
                s=s+list(gen_dump(line))
          f.write("""
void %s::print_data(const std::string & name)
{
    fn=name+"_d.cpp";
    FILE * fp=fopen(fn.c_str(),"w");
    if(fp)
    {  
"""%(l[0]))
          # scan  vector<int/float/pair>,blob<int/dtype>,map<int,string>
          for type,v in s:
        	   if type=='vector<int>':
        	   	   f.write("print_vector_int_data(fp,\"%s\",%s);\n"%(v,v))
        	   	   continue
        	   if type=='vector<float>':
        	   	   f.write("print_vector_float_data(fp,\"%s\",%s);\n"%(v,v))
        	   	   continue        	   	    
        	   if type=='vector<string>':
        	   	   f.write("print_vector_string_data(fp,\"%s\",%s);\n"%(v,v))
        	   	   continue        	   	    
        	   if type=='Blob<int>':
        	   	   f.write("print_blob_int_data(fp,\"%s\",%s);\n"%(v,v))
        	   	   f.write("print_blob_int_shape_data(fp,\"%s\",%s);\n"%(v,v))	   	   
        	   	   continue        	   	    
        	   if type=='Blob<Dtype>':
        	   	   f.write("print_blob_dtype_data(fp,\"%s\",%s);\n"%(v,v))
        	   	   f.write("print_blob_dtype_shape_data(fp,\"%s\",%s);\n"%(v,v))	   	   
        	   	   continue        	   	    
        	   if type=='const vector<int>*':
        	   	   f.write("print_vector_int_ptr_data(fp,\"%s\",%s);\n"%(v,v))
        	   	   continue        	   	    
        	   if type=='map<int,string>' or type=='map<int, string>':
        	   	   f.write("print_map_int_string_data(fp,\"%s\",%s);\n"%(v,v))
        	   	   continue        	   	    
        	   if type=='vector<pair<int, int> >':
        	   	   f.write("print_vector_pair_int_int_data(fp,\"%s\",%s);\n"%(v,v))
        	   	   continue      
        	   if type=='shared_ptr<DataTransformer<Dtype> >':
        	   	   f.write("print_data_transformer_data(fp,\"%s\", %s);\n"%(v,v))
        	   	   continue  	           	   	    

          f.write("\nprintf(fp,\"struct %s_conf %%s = {\\n\",name.c_str());\n"%(l[0]))          
          first=True
          for type,v in s:
                   if not first:
                           f.write('fprintf(fp,",\\n");\n')
                   first=False
        	   if type=='vector<int>':
        	   	   f.write("print_vector_int(fp,\"%s\",%s);\n"%(v,v))
        	   	   continue
        	   if type=='vector<float>':
        	   	   f.write("print_vector_float(fp,\"%s\",%s);\n"%(v,v))
        	   	   continue        	   	    
        	   if type=='vector<string>':
        	   	   f.write("print_vector_string(fp,\"%s\",%s);\n"%(v,v))
        	   	   continue        	   	    
        	   if type=='Blob<int>':
        	   	   f.write("print_blob_int(fp,\"%s\",%s);\n"%(v,v))
        	   	   continue        	   	    
        	   if type=='Blob<Dtype>':
        	   	   f.write("print_blob_dtype(fp,\"%s\",%s);\n"%(v,v))
        	   	   continue        	   	    
        	   if type=='const vector<int>*':
        	   	   f.write("print_vector_int_ptr(fp,\"%s\",%s);\n"%(v,v))
        	   	   continue        	   	    
        	   if type=='map<int,string>' or type=='map<int, string>':
        	   	   f.write("print_map_int_string(fp,\"%s\",%s);\n"%(v,v))
        	   	   continue        	   	    
        	   if type=='vector<pair<int, int> >':
        	   	   f.write("print_vector_pair_int_int(fp,\"%s\",%s);\n"%(v,v))
        	   	   continue      
        	   if type=='shared_ptr<DataTransformer<Dtype> >':
        	   	   f.write("print_data_transformer(fp,\"%s\", %s);\n"%(v,v))
        	   	   continue         
        	   f.write("print(fp,\"%s\",%s);\n"%(v,v))

          f.write("fprintf(fp,\"\\n};\\n\");\n")           
          f.write("fclose(fp);\n   }\n}")      
  f.close()
