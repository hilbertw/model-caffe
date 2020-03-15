import os
import sys
import caffe
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
  	f.write("dump_data(fp,\"%s\",%s);\n"%(s,s))

caffe_path="caffe"

output_path=caffe_path+"/include/hack/"
if not os.path.exists(output_path):
    os.makedirs(output_path)
     
for l in layer_list:
        if l[1]!="":
            fn="%s_dump.h"%(l[0])
            with open(output_path+fn,"w") as f:
    	        f.write("""
virtual void dump(const std::string & name)const
{
    std::string fn = name+std::string("_dump.txt");
    FILE * fp =fopen(fn.c_str(),"w");
    if(fp)
    {
""")
    	        lines=l[1].strip().split(";")
    	        for line in lines:
#                 print(line)         
                 line.strip(' \n')         
                 line.rstrip(' \n')          
#                 print(line)
                 if line !="":
                   gen_dump(f,line)
    	        f.write("""
       fclose(fp);
    }
}
""")
    	        f.close()



if len(sys.argv)<3:
    sys.exit(0)

model=sys.argv[1]
weights=sys.argv[2]

print(model)
print(weights)


net = caffe.Net(model, caffe.TEST)
net.copy_from(weights)

#print_net(net)

with open("gen/net_dump.cpp","w") as f:
     f.write(
"""
#include "sc_net.h"
void sc_net::dump()
{
"""
     )

     for name in net._layer_names:
         f.write("%s.get().dump(\"sc_%s\");\n"%(name,name))
     f.write(
"""
}
""")
     f.close()
