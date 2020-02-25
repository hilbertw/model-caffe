
base_convolution="""  
  Blob<int> kernel_shape_; 
  Blob<int> stride_;  
  Blob<int> pad_;  
  Blob<int> dilation_;  
  Blob<int> conv_input_shape_;
  vector<int> col_buffer_shape_; 
  vector<int> output_shape_;
  const vector<int>* bottom_shape_;

  int num_spatial_axes_;
  int bottom_dim_;
  int top_dim_;

  int channel_axis_;
  int num_;
  int channels_;
  int group_;
  int out_spatial_dim_;
  int weight_offset_;
  int num_output_;
  bool bias_term_;
  bool is_1x1_;
  bool force_nd_im2col_;
  int num_kernels_im2col_;
  int num_kernels_col2im_;
  int conv_out_channels_;
  int conv_in_channels_;
  int conv_out_spatial_dim_;
  int kernel_dim_;
  int col_offset_;
  int output_offset_;

  Blob<Dtype> col_buffer_;
  Blob<Dtype> bias_multiplier_;  
"""


concat="""
  int count_;
  int num_concats_;
  int concat_input_size_;
  int concat_axis_;
"""

conv=base_convolution+"""
"""

detect_output="""
  int num_classes_;
  bool share_location_;
  int num_loc_classes_;
  int background_label_id_;
  CodeType code_type_;
  bool variance_encoded_in_target_;
  int keep_top_k_;
  float confidence_threshold_;

  int num_;
  int num_priors_;

  float nms_threshold_;
  int top_k_;
  float eta_;

  bool need_save_;
  string output_directory_;
  string output_name_prefix_;
  string output_format_;
  map<int, string> label_to_name_;
  map<int, string> label_to_display_name_;
  vector<string> names_;
  vector<pair<int, int> > sizes_;
  int num_test_image_;
  int name_count_;
  bool has_resize_;
  ResizeParameter resize_param_;

  ptree detections_;

  bool visualize_;
  float visualize_threshold_;
  shared_ptr<DataTransformer<Dtype> > data_transformer_;
  string save_file_; 
  Blob<Dtype> bbox_preds_;
  Blob<Dtype> bbox_permute_;
  Blob<Dtype> conf_permute_;
 
"""

flatten="""
"""

input="""
"""

normalize="""
  Blob<Dtype> norm_;
  Blob<Dtype> sum_channel_multiplier_, sum_spatial_multiplier_;
  Blob<Dtype> buffer_, buffer_channel_, buffer_spatial_;
  bool across_spatial_;
  bool channel_shared_;
  Dtype eps_;
"""
  
permute="""
  int num_axes_;
  bool need_permute_;  
  Blob<int> permute_order_;
  Blob<int> old_steps_;
  Blob<int> new_steps_;
"""


pooling="""  
  int kernel_h_, kernel_w_;
  int stride_h_, stride_w_;
  int pad_h_, pad_w_;
  int channels_;
  int height_, width_;
  int pooled_height_, pooled_width_;
  bool global_pooling_;
  Blob<Dtype> rand_idx_;
  Blob<int> max_idx_;
"""
  
prior_box="""  
  vector<float> min_sizes_;
  vector<float> max_sizes_;
  vector<float> aspect_ratios_;
  bool flip_;
  int num_priors_;
  bool clip_;
  vector<float> variance_;
  int img_w_;
  int img_h_;
  float step_w_;
  float step_h_;

  float offset_;
"""
    
relu="""
"""
  
reshape="""  
  vector<int> copy_axes_;
  int inferred_axis_;
  int constant_count_;
"""
  
softmax="""
  int outer_num_;
  int inner_num_;
  int softmax_axis_;
  Blob<Dtype> sum_multiplier_;
  Blob<Dtype> scale_;  
"""
  
split="""
  int count_;  
"""
  
layer_list=[
["BaseConvolutionLayer",base_convolution],
["ConcatLayer",concat],
["ConvolutionLayer",conv],
["DetectionOutputLayer",detect_output],
["FlattenLayer",flatten],
["InputLayer",input],
["NormalizeLayer",normalize],
["PermuteLayer",permute],
["PoolingLayer",pooling],
["PriorBoxLayer",prior_box],
["ReLULayer",relu],
["ReshapeLayer",reshape],
["SoftmaxLayer",softmax],
["SplitLayer",split]
]
