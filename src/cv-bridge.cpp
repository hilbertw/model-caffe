#include "ssd_detect.h"
#include "cv-bridge.h"

namespace bridge{

cv:: Mat img;
Detector * detector; 
int init(caffe::Blob<float>* input_blob,caffe::Blob<float>* output_blob)
{

    detector=new  Detector(input_blob,output_blob);    
    return detector==NULL?-1:0;
}

int read_in_image(const char *fn)
{
   img = cv::imread(fn,1);
   if(detector)
   detector->Detect_in(img);
   return 0;
}

int read_out_result()
{
   std::vector<vector<float> >  detections = detector->Detect_out();
    
   for (unsigned int i = 0; i < detections.size(); ++i) {
        const vector<float>& d = detections[i];
        // Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
        CHECK_EQ(d.size(), 7);
        const float score = d[2];
        const int tpye = d[1];
        if (tpye == 15) {
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
            cv::rectangle(img, pt1, pt2, cvScalar(0, 255, 0), 3, 8, 0);
       }
   }
   cv::imshow("img",img);

   cv::waitKey(0);
   return 0;
}

}
