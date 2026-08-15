// Diagnostic executable that exposes OFIQ's original-image landmarks and Sharpness result.
#include "image_io.h"
#include "ofiq_lib.h"

#include <cstdint>
#include <iostream>
#include <memory>
#include <string>

int main(int argc, char** argv)
{
    if (argc != 4)
    {
        std::cerr << "usage: ofiq_preprocessing DATA_DIR CONFIG_FILE IMAGE\n";
        return 2;
    }

    const auto implementation = OFIQ::Interface::getImplementation();
    auto status = implementation->initialize(argv[1], argv[2]);
    if (status.code != OFIQ::ReturnCode::Success)
    {
        std::cerr << "initialize: " << status.info << '\n';
        return 3;
    }

    OFIQ::Image image;
    status = OFIQ_LIB::readImage(argv[3], image);
    if (status.code != OFIQ::ReturnCode::Success)
    {
        std::cerr << "read: " << status.info << '\n';
        return 4;
    }

    OFIQ::FaceImageQualityAssessment assessment;
    OFIQ::FaceImageQualityPreprocessingResult preprocessing;
    status = implementation->vectorQualityWithPreprocessingResults(
        image,
        assessment,
        preprocessing,
        static_cast<uint32_t>(OFIQ::PreprocessingResultType::Landmarks));
    if (status.code != OFIQ::ReturnCode::Success)
    {
        std::cerr << "assess: " << status.info << '\n';
        return 5;
    }

    const auto sharpness = assessment.qAssessments.at(OFIQ::QualityMeasure::Sharpness);
    std::cout << "image;" << image.width << ';' << image.height << '\n';
    std::cout << "face;" << assessment.boundingBox.xleft << ';'
              << assessment.boundingBox.ytop << ';' << assessment.boundingBox.width << ';'
              << assessment.boundingBox.height << '\n';
    std::cout << "sharpness;" << sharpness.rawScore << ';' << sharpness.scalar << ';'
              << static_cast<int>(sharpness.code) << '\n';
    for (std::size_t index = 0; index < preprocessing.m_landmarks.landmarks.size(); ++index)
    {
        const auto& point = preprocessing.m_landmarks.landmarks[index];
        std::cout << "landmark;" << index << ';' << point.x << ';' << point.y << '\n';
    }
    return 0;
}
