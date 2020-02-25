#include "ssd_detect.h"

DEFINE_string(mean_file, "",
    "The mean file used to subtract from the input image.");
DEFINE_string(mean_value, "104,117,123",
    "If specified, can be one value or can be same as image channels"
    " - would subtract from the corresponding channel). Separated by ','."
    "Either mean_file or mean_value should be provided, not both.");
DEFINE_string(img_file, "test.png",
    "input image.");
DEFINE_double(confidence_threshold, 0.01,
    "Only store detections with score higher than the threshold.");

using namespace std;

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  // Print output to stderr (while still logging)
  FLAGS_alsologtostderr = 1;

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Do detection using SSD mode.\n"
        "Usage:\n"
        "    ssd_detect [FLAGS] model_file weights_file\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc < 3) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "examples/ssd/ssd_detect");
    return 1;
  }
  const string& model_file = argv[1];
  const string& weights_file = argv[2];
  const string& mean_file = FLAGS_mean_file;
  const string& mean_value = FLAGS_mean_value;
  const string& img_file = FLAGS_img_file;
  const float confidence_threshold = FLAGS_confidence_threshold;

  // Initialize the network.
  Detector detector(model_file, weights_file, mean_file, mean_value);


  cout << "Press ESC/Q on Windows to terminate" << endl;

  //cv::namedWindow("img", CV_WINDOW_NORMAL);
  //cv::setWindowProperty("img", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
#ifdef USE_CAM
  // Create a VideoCapture object and use camera to capture the video
  VideoCapture cap(0);

  // Check if camera opened successfully
  if(!cap.isOpened())
  {
    cout << "Error opening video stream" << endl;
    return -1;
  }
  while (1) {
      cv::Mat img;
      cap >> img;
#else
      cv::Mat img = cv::imread(img_file);
      //cv::Mat img = cv::imread(img_file, CV_LOAD_IMAGE_COLOR);
#endif
      std::vector<vector<float> > detections = detector.Detect(img);
      /* Print the detection results. */
      for (unsigned int i = 0; i < detections.size(); ++i) {
        const vector<float>& d = detections[i];
        // Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
        CHECK_EQ(d.size(), 7);
        const float score = d[2];
        const int tpye = d[1];
        if (tpye == 15) {
          if (score >= confidence_threshold) {
            // cout << static_cast<int>(d[1]) << endl;
            // out << file << " ";
            // out << static_cast<int>(d[1]) << " ";
            // out << score << " ";
            // out << static_cast<int>(d[3] * img.cols) << " ";
            // out << static_cast<int>(d[4] * img.rows) << " ";
            // out << static_cast<int>(d[5] * img.cols) << " ";
            // out << static_cast<int>(d[6] * img.rows) << std::endl;
            cv::Point pt1, pt2;
            pt1.x = (img.cols*d[3]);
            pt1.y = (img.rows*d[4]);
            pt2.x = (img.cols*d[5]);
            pt2.y = (img.rows*d[6]);
            if (pt1.x > -img.cols && pt1.y > -img.rows && pt2.x > -img.cols && pt2.y > -img.rows
                && pt1.x < 2*img.cols && pt1.y < 2*img.rows && pt2.x < 2*img.cols && pt2.y< 2*img.rows) {
              cv::rectangle(img, pt1, pt2, cvScalar(0, 255, 0), 3, 8, 0);
            }
          }
        }
      } 
      cv::imshow("img", img);
#ifdef USE_CAM
      // Press  ESC on keyboard to  exit
     char c = (char)waitKey(1);
     if( c == 27 ) 
        break;
  }
  
  // When everything done, release the video capture and write object
  cap.release();
#else
      cv::waitKey(0);
#endif
  cv::destroyAllWindows();
  return 0;
}
