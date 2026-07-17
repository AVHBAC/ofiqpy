#!/usr/bin/env bash
# Build the OPTIONAL ctypes bridges against OFIQ's own conan-built OpenCV static libs.
#
# These are DIAGNOSTIC/VERIFICATION tools only — they let you confirm that OFIQ's compiled
# estimateAffinePartial2D / warpAffine / resize / SSD-dnn-forward are bit-identical to the
# pip opencv-python wheel. The ofiqpy runtime does NOT need them (it uses pip cv2).
#
# Requires an OFIQ conan build on disk. Set OCV to the OpenCV package prefix and (for the
# SSD bridge) ZLIB / PROTO to the zlib / protobuf package prefixes, e.g.:
#   OCV=~/.conan2/p/b/opencXXXX/p ZLIB=~/.conan2/p/b/zlibXXXX/p PROTO=~/.conan2/p/b/protoXXXX/p ./build.sh
set -euo pipefail
cd "$(dirname "$0")"

: "${OCV:?set OCV to the conan OpenCV package prefix (contains lib/ and include/opencv4)}"
INC="$OCV/include/opencv4"
LIBS_CORE=("$OCV/lib/libopencv_calib3d.a" "$OCV/lib/libopencv_features2d.a" \
           "$OCV/lib/libopencv_flann.a" "$OCV/lib/libopencv_imgproc.a" "$OCV/lib/libopencv_core.a")

echo "building ofiq_cv.so (estimateAffinePartial2D / warpAffine / resize)"
g++ -O2 -fPIC -shared ofiq_cv.cpp -o ofiq_cv.so -I"$INC" \
    "${LIBS_CORE[@]}" -L"${ZLIB:-/usr}/lib" -lz -lpthread -ldl -lm

if [[ -n "${PROTO:-}" ]]; then
  echo "building ofiq_ssd.so (SSD dnn forward)"
  g++ -O2 -fPIC -shared ofiq_ssd.cpp -o ofiq_ssd.so -I"$INC" \
      "$OCV/lib/libopencv_dnn.a" "$OCV/lib/libopencv_imgproc.a" "$OCV/lib/libopencv_core.a" \
      "$PROTO"/lib/*.a -L"${ZLIB:-/usr}/lib" -lz -lpthread -ldl -lm
else
  echo "skipping ofiq_ssd.so (set PROTO to build it)"
fi
echo "done."
