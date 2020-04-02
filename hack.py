import caffe

import numpy as np
import argparse
import os
import sys


if len(sys.argv)<3:
    sys.exit(0)

model=sys.argv[1]
weights=sys.argv[2]

print(model)
print(weights)


net = caffe.Net(model, caffe.TEST)
net.copy_from(weights)

print(dir(net))

net.hack()
