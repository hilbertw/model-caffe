from layer_list import layer_list

def split_type(s):
   words=s.rsplit(' ',1)
   return words[0],words[1]

def gen_dump(f,s):
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
  	f.write("print(fp,\"%s\",%s);\n"%(s,s))
     
with open("layer_dump.cpp","w") as f:
    for l in layer_list:
        if l[1]!="":
    	        f.write("void %s::dump(FILE * fp)\n{\n"%(l[0]))
    	        lines=l[1].strip().split(";")
    	        for line in lines:
#                 print(line)         
                 line.strip(' \n')         
                 line.rstrip(' \n')          
#                 print(line)
                 if line !="":
                   gen_dump(f,line)
    	        f.write("\n}\n")      
    f.close()