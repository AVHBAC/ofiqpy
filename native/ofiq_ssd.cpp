// Bridge to OFIQ's OpenCV dnn for the SSD forward pass (Caffe).
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include <cstring>

static cv::dnn::Net g_net;
static bool g_loaded = false;

extern "C" {

void ssd_load(const char* prototxt, const char* caffemodel) {
    g_net = cv::dnn::readNetFromCaffe(prototxt, caffemodel);
    g_loaded = true;
}

// padded BGR uint8 -> blobFromImage(300, mean 104/117/123, swapRB=false) -> forward.
// out: rows of [conf,l,t,r,b] (5 floats each). returns n rows (<=max_n).
int ssd_forward(const unsigned char* img, int h, int w, float* out, int max_n) {
    if (!g_loaded) return -1;
    cv::Mat src(h, w, CV_8UC3, (void*)img);
    cv::Mat blob = cv::dnn::blobFromImage(src, 1.0, cv::Size(300, 300),
                                          cv::Scalar(104, 117, 123), false, false);
    g_net.setInput(blob);
    cv::Mat det = g_net.forward();  // 1x1xNx7
    const int N = det.size[2];
    const float* p = (const float*)det.data;
    int n = 0;
    for (int i = 0; i < N && n < max_n; ++i) {
        const float* row = p + i * 7;
        out[n*5+0] = row[2]; out[n*5+1] = row[3]; out[n*5+2] = row[4];
        out[n*5+3] = row[5]; out[n*5+4] = row[6];
        ++n;
    }
    return n;
}

}
