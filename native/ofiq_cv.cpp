// Bridge to OFIQ's own conan-built OpenCV 4.5.5 (gcc9) for bit-identical alignment.
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <cstring>
#include <vector>

extern "C" {

// LMEDS partial-affine (similarity). src/dst: n*2 doubles. outM: 6 doubles (2x3). ret 0=ok.
int estimate_affine_partial2d(const double* src, const double* dst, int n, double* outM) {
    std::vector<cv::Point2f> s, d;
    s.reserve(n); d.reserve(n);
    for (int i = 0; i < n; ++i) {
        s.emplace_back((float)src[2*i], (float)src[2*i+1]);
        d.emplace_back((float)dst[2*i], (float)dst[2*i+1]);
    }
    cv::Mat M = cv::estimateAffinePartial2D(s, d, cv::noArray(), cv::LMEDS);
    if (M.empty()) return 1;
    for (int r = 0; r < 2; ++r)
        for (int c = 0; c < 3; ++c)
            outM[r*3+c] = M.at<double>(r, c);
    return 0;
}

// warpAffine BGR uint8, default INTER_LINEAR + BORDER_CONSTANT 0. M: 6 doubles (2x3).
void warp_affine(const unsigned char* img, int h, int w, int ch,
                 const double* M, unsigned char* out, int oh, int ow) {
    cv::Mat src(h, w, ch == 3 ? CV_8UC3 : CV_8UC1, (void*)img);
    cv::Mat mm(2, 3, CV_64F, (void*)M);
    cv::Mat dst;
    cv::warpAffine(src, dst, mm, cv::Size(ow, oh));
    std::memcpy(out, dst.data, (size_t)oh * ow * ch);
}

// resize INTER_LINEAR uint8.
void resize_linear(const unsigned char* img, int h, int w, int ch,
                   unsigned char* out, int oh, int ow) {
    cv::Mat src(h, w, ch == 3 ? CV_8UC3 : CV_8UC1, (void*)img);
    cv::Mat dst;
    cv::resize(src, dst, cv::Size(ow, oh), 0, 0, cv::INTER_LINEAR);
    std::memcpy(out, dst.data, (size_t)oh * ow * ch);
}

}
