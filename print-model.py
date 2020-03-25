import caffe

import numpy as np
import argparse
import os
import sys


def find_in_bottom(net,id):
    i=0
    for l in net.layers:
        ids=net._bottom_ids(i)
        if id in ids:
           return i
        i=i+1
    return -1

def find_in_top(net,id):
    i=0
    for l in net.layers:
        ids=net._top_ids(i)
        if id in ids:
           return i
        i=i+1
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
    for id in net._top_ids(i):
        j=find_in_bottom(net,id)
        if j>=0:
            n.append(net._layer_names[j])

    m.append(','.join(n))
    i=i+1
i=0

with open("model.txt","w") as f:
   for n in m:
        name=net._layer_names[i]
        f.write('\n%s:\n'%(name))
        f.write('[%s]\n'%(n))  
        i=i+1
   f.close()

