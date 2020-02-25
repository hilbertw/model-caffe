import sys
import os


input=[]
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
       input.append(words)
       
       
f.close()

       
with open("gen/objs.mak","w") as ff:

    ff.write("EXT_OBJS=\\\n")
    for l in input:      
       ff.write("\t%s_ext.o \\\n"%(l[1])) 
    ff.write("\n")       

    ff.write("OBJS=\\\n")    
    for l in input:      
       ff.write("\t%s.o \\\n"%(l[1]))        
    ff.write("\n")
           
    ff.close()
