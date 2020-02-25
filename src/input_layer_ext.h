
#include "caffe/layers/input_layer.hpp"

class InputLayer_ext: public caffe::InputLayer< float > {
public:
InputLayer_ext();
virtual ~InputLayer_ext() {}

//void setup(permute_param  &param);
};
