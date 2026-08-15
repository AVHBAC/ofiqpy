// Diagnostic bridge to an operator-supplied OFIQ OpenCV 4.5.5 build.
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ml.hpp>
#include <array>
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

// OFIQ Sharpness.cpp's 26 focus features on a caller-supplied grayscale crop/mask.
int sharpness_features(const unsigned char* gray, const unsigned char* mask,
                       int h, int w, double* out) {
    if (gray == nullptr || mask == nullptr || out == nullptr || h <= 0 || w <= 0)
        return 1;
    cv::Mat grayImage(h, w, CV_8UC1, const_cast<unsigned char*>(gray));
    cv::Mat maskImage(h, w, CV_8UC1, const_cast<unsigned char*>(mask));
    cv::Mat grayBlur3;
    cv::GaussianBlur(grayImage, grayBlur3, cv::Size(3, 3), 0);

    int index = 0;
    const auto appendMeanStdDev = [&](const cv::Mat& values) {
        cv::Mat mean;
        cv::Mat stddev;
        cv::meanStdDev(values, mean, stddev, maskImage);
        out[index++] = mean.at<double>(0, 0);
        out[index++] = stddev.at<double>(0, 0);
    };

    const std::array<int, 5> kernelSizes{1, 3, 5, 7, 9};
    for (const int k : kernelSizes) {
        cv::Mat laplacian;
        cv::Laplacian(grayBlur3, laplacian, CV_64F, k);
        appendMeanStdDev(cv::abs(laplacian));
    }
    for (const int k : std::array<int, 3>{3, 5, 7}) {
        cv::Mat grayMeanBlur;
        cv::Mat absdiff;
        cv::blur(grayImage, grayMeanBlur, cv::Size(k, k));
        cv::absdiff(grayImage, grayMeanBlur, absdiff);
        appendMeanStdDev(absdiff);
    }
    for (const int k : kernelSizes) {
        cv::Mat sobel;
        cv::Sobel(grayImage, sobel, CV_64F, 1, 1, k);
        appendMeanStdDev(cv::abs(sobel));
    }
    return index == 26 ? 0 : 2;
}

// Evaluate OFIQ's RTrees artifact through the same conan OpenCV build as OFIQ.
int sharpness_rtree_raw(const char* modelPath, const float* features,
                        int featureCount, float* raw, int* numTrees) {
    if (modelPath == nullptr || features == nullptr || raw == nullptr ||
        numTrees == nullptr || featureCount != 26)
        return 1;
    const auto rtree = cv::ml::RTrees::load(modelPath);
    if (rtree.empty())
        return 2;
    const cv::Mat featureRow(1, featureCount, CV_32F,
                             const_cast<float*>(features));
    cv::Mat prediction;
    rtree->predict(featureRow, prediction, cv::ml::StatModel::RAW_OUTPUT);
    *raw = prediction.at<float>(0, 0);
    *numTrees = rtree->getTermCriteria().maxCount;
    return 0;
}

// ADNet's cv::Mat::convertTo(alpha=2/255, beta=-1) plus HWC-to-CHW loop.
int adnet_input(const unsigned char* bgr, int h, int w, float* out) {
    if (bgr == nullptr || out == nullptr || h <= 0 || w <= 0)
        return 1;
    const cv::Mat image(h, w, CV_8UC3, const_cast<unsigned char*>(bgr));
    const cv::Mat flattened = image.reshape(1, 1);
    std::vector<float> converted;
    flattened.convertTo(converted, CV_32FC1, 2. / 255, -1.);
    std::size_t index = 0;
    for (std::size_t channel = 0; channel < 3; ++channel)
        for (std::size_t pixel = channel; pixel < converted.size(); pixel += 3)
            out[index++] = converted[pixel];
    return index == static_cast<std::size_t>(h * w * 3) ? 0 : 2;
}

// OpenCV float conversion/normalization/resize followed by HWC-to-CHW.
int normalized_chw(const unsigned char* bgr, int h, int w, int outH, int outW,
                   int swapRB, double divisor,
                   double mean0, double mean1, double mean2,
                   double std0, double std1, double std2, float* out) {
    if (bgr == nullptr || out == nullptr || h <= 0 || w <= 0 ||
        outH <= 0 || outW <= 0 || divisor == 0.0 ||
        std0 == 0.0 || std1 == 0.0 || std2 == 0.0)
        return 1;
    const cv::Mat source(h, w, CV_8UC3, const_cast<unsigned char*>(bgr));
    cv::Mat transformed;
    if (swapRB)
        cv::cvtColor(source, transformed, cv::COLOR_BGR2RGB);
    else
        transformed = source;
    transformed.convertTo(transformed, CV_32FC3);
    transformed /= cv::Scalar(divisor, divisor, divisor);
    transformed -= cv::Scalar(mean0, mean1, mean2);
    transformed /= cv::Scalar(std0, std1, std2);
    if (transformed.rows != outH || transformed.cols != outW)
        cv::resize(transformed, transformed, cv::Size(outW, outH), 0, 0, cv::INTER_LINEAR);

    const int channelSize = outH * outW;
    for (int index = 0; index < channelSize; ++index) {
        const auto& pixel = transformed.at<cv::Vec3f>(index / outW, index % outW);
        for (int channel = 0; channel < 3; ++channel)
            out[index + channel * channelSize] = pixel[channel];
    }
    return 0;
}

}
