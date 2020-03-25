include gen/objs.mak
include gen/w_objs.mak
include gen/config_objs.mak

#$info $(OBJS))
#$info $(EXT_OBJS))

SC_OBJS:= main.o sc_net.o caffe.pb.o layer_params.o  setup_layers.o config_layers.o layer_config.o  sc_layer.o net_blobs.o net_weights.o net_wire.o
SC_OBJS:= $(SC_OBJS) cv-bridge.o ssd_detect.o conv_struct.o net_dump.o util.o net_debug.o 

CAFFE_SOURCE=/disk1/model-caffe/caffe
SC_SOURCE=/disk1/model-caffe/
CFLAGS:=-g -DCPU_ONLY -DSC_DISABLE_VIRTUAL_BIND -DSC_BUILD
CFLAGS:=$(CFLAGS) -m64 -I/usr/local/include -I/usr/local/include/opencv
LDFLAGS :=-g $(LDFLAGS) -m64
SYSTEMC_INSTALL:= /disk1/systemc/install

CFLAGS:=$(CFLAGS) --std=c++11  -I $(CAFFE_SOURCE)/include -I$(SC_SOURCE)/gen -I$(SC_SOURCE)/src -I$(SYSTEMC_INSTALL)/include -I$(CAFFE_SOURCE)/build/src
LDFLAGS:=$(LDFLAGS) -L. -L$(SYSTEMC_INSTALL)/lib  -L $(CAFFE_SOURCE)/build/lib -lsystemc -lcaffe -lprotobuf  -lglog -lm

BOOST_LIBS:=-lboost_system -lboost_filesystem -lboost_regex
#OPENCV_LIBS:= -L /usr/local/lib -lopencv_calib3d -lopencv_core -lopencv_dnn -lopencv_features2d -lopencv_flann -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_ml -lopencv_objdetect -lopencv_photo -lopencv_shape -lopencv_stitching -lopencv_superres -lopencv_video -lopencv_videoio -lopencv_videostab


OPENCV_LIBS:= -lopencv_videostab -lopencv_video -lopencv_ts -lopencv_superres -lopencv_stitching -lopencv_photo -lopencv_ocl -lopencv_objdetect -lopencv_ml -lopencv_legacy -lopencv_imgproc -lopencv_highgui -lopencv_gpu -lopencv_flann -lopencv_features2d -lopencv_core -lopencv_contrib -lopencv_calib3d

ALL_OBJS:=$(addprefix obj/, $(EXT_OBJS) $(SC_OBJS) $(WEIGHT_OBJS) $(CONF_OBJS))

all: sim 

obj/caffe.pb.o:$(CAFFE_SOURCE)/build/src/caffe/proto/caffe.pb.cc
	 g++ -c -o $@ $(CFLAGS)  $<

obj/%.o:$(CAFFE_SOURCE)/src/caffe/layers/%.cpp
	 g++ -c -o $@ $(CFLAGS)  $<

obj/%.o:$(SC_SOURCE)/gen/%.cpp
	 g++ -c -o $@ $(CFLAGS)  $<
	 
obj/%.o:$(SC_SOURCE)/gen/%.cc
	 g++ -c -o $@ $(CFLAGS)  $<

obj/%.o:$(SC_SOURCE)/src/%.cpp
	 g++ -c -o $@ $(CFLAGS)  $<

sim:$(ALL_OBJS)
	g++ -o $@ $^  $(LDFLAGS) $(BOOST_LIBS) $(OPENCV_LIBS) 


run:
        LD_LIBRARY_PATH=$(SYSTEMC_INSTALL)/lib ./sim
	

.phony: clean

clean:
	rm -rf sim obj/*	 
	 

