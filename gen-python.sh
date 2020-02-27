#!/bin/bash

echo  ${PWD}/caffe/src/caffe/proto/
protoc --python_out=. -I${PWD} -I${PWD}/caffe/src/caffe/proto caffe.proto 

 
