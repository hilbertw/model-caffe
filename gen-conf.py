from layer_list import layer_list

def split_type(s):
   words=s.rsplit(' ',1)
   return words[0],words[1]

def print_field(f,type,v):
	   if type=='vector<int>':
	   	   f.write("vector_int_def %s;\n"%( v))
	   	   return
	   if type=='vector<float>':
	   	   f.write("vector_float_def %s;\n"%( v))
	   	   return        	   	    
	   if type=='vector<string>':
	   	   f.write("vector_string_def %s;\n"%( v))
	   	   return        	   	    
	   if type=='Blob<int>':
	   	   f.write("blob_int_def %s;\n"%( v))  	   
	   	   return        	   	    
	   if type=='Blob<Dtype>':
	   	   f.write("blob_dtype_def %s;\n"%( v))
	   	   return        	   	    
	   if type=='const vector<int>*':
	   	   f.write("vector_int_ptr_def %s;\n"%( v))
	   	   return        	   	    
	   if type=='map<int,string>' or type=='map<int, string>':
	   	   f.write("map_int_string_def %s;\n"%( v))
	   	   return        	   	    
	   if type=='vector<pair<int, int> >':
	   	   f.write("vector_pair_int_int_def %s;\n"%( v))
	   	   return  	   	   
	   if type=='shared_ptr<DataTransformer<Dtype> >':
	   	   f.write("data_transformer_def %s;\n"%( v))
	   	   return  	   	
	   f.write("%s %s;\n"%(type,v))    
     
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
  	 
with open("layer_conf.h","w") as f:
  f.write("""
#pragma once
#include "types.h"
	""")
  for l in layer_list:
    if l[1]!="":
      f.write("struct %s_conf\n{\n"%(l[0]))
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