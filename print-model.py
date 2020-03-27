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



if len(sys.argv)<3:
    sys.exit(0)

model=sys.argv[1]
weights=sys.argv[2]

print(model)
print(weights)


net = caffe.Net(model, caffe.TEST)
net.copy_from(weights)

m=[]
i=0
for l in net.layers:
    n=[]
    n1=[]
    for id in net._top_ids(i):
        j=find_in_bottom(net,id,i)
        if j>=0:
            n.append(net._layer_names[j])
        n1.append(net._blob_names[id])
    n2=[]
    n3=[]
    for id in net._bottom_ids(i):
        j=find_in_top(net,id,i)
        if j>=0:
            n2.append(net._layer_names[j])
        n3.append(net._blob_names[id])

    m.append((','.join(n),','.join(n2),','.join(n1),','.join(n3)))
    i=i+1

i=0

with open("model.txt","w") as f:
   for n,n2,n1,n3 in m:
        name=net._layer_names[i]
        f.write('\n%s:\n'%(name))
        f.write('bottom:[%s]\n'%(n2))  
        f.write('top:[%s]\n'%(n))  
        f.write('bottom blob:[%s]\n'%(n3))  
        f.write('top blob:[%s]\n'%(n1))  
        i=i+1
   f.close()

