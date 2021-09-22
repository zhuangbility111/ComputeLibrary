g++ -o build/arm64/examples/neon_cnn.o -c -Wall -DARCH_ARM -Wextra -pedantic -Wdisabled-optimization -Wformat=2 -Winit-self -Wstrict-overflow=2 -Wswitch-default -std=c++14 -Woverloaded-virtual -Wformat-security -Wctor-dtor-privacy -Wsign-promo -Weffc++ -Wno-overlength-strings -Wlogical-op -Wnoexcept -Wstrict-null-sentinel -C -march=armv8.2-a+sve+fp16+dotprod -DENABLE_SVE -DARM_COMPUTE_ENABLE_SVE -Wno-ignored-attributes -DENABLE_FP16_KERNELS -DENABLE_FP32_KERNELS -DENABLE_QASYMM8_KERNELS -DENABLE_QASYMM8_SIGNED_KERNELS -DENABLE_QSYMM16_KERNELS -DENABLE_INTEGER_KERNELS -DENABLE_NHWC_KERNELS -DENABLE_NCHW_KERNELS -O3 -mcpu=a64fx -D_GLIBCXX_USE_NANOSLEEP -DARM_COMPUTE_CPP_SCHEDULER=1 -DARM_COMPUTE_ENABLE_FP16 -DARM_COMPUTE_CPU_ENABLED -Dic=$1 -Doc=$2 -Dow=$3 -Iinclude -I. -I. examples/neon_cnn.cpp

g++ -o build/arm64/examples/neon_cnn -mcpu=a64fx build/arm64/examples/neon_cnn.o build/arm64/utils/Utils.o -Lbuild/arm64 -L. -lpthread -larm_compute -larm_compute_core

./build/arm64/examples/neon_cnn $4 > data.tmp

tail -n 8 data.tmp
