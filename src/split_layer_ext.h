
#include "caffe/layers/split_layer.hpp"

class SplitLayer_ext: public caffe::SplitLayer< float > {
public:
SplitLayer_ext();
virtual ~SplitLayer_ext() {}

//void setup(permute_param  &param);
};
