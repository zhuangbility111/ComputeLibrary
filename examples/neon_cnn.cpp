/*
 * Copyright (c) 2016-2020 Arm Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include "arm_compute/runtime/NEON/NEFunctions.h"

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/Allocator.h"
#include "arm_compute/runtime/BlobLifetimeManager.h"
#include "arm_compute/runtime/MemoryManagerOnDemand.h"
#include "arm_compute/runtime/PoolManager.h"
#include "arm_compute/runtime/Scheduler.h"
#include "utils/Utils.h"
#include "timer.h"

#ifndef ic
    #define ic 512
#endif

#ifndef oc
    #define oc 512
#endif

#ifndef ow
    #define ow 14
#endif

using namespace arm_compute;
using namespace utils;

class NEONCNNExample : public Example
{
public:
    bool do_setup(int argc, char **argv) override
    {
        ARM_COMPUTE_UNUSED(argc);
        ARM_COMPUTE_UNUSED(argv);

        // Create memory manager components
        // We need 2 memory managers: 1 for handling the tensors within the functions (mm_layers) and 1 for handling the input and output tensors of the functions (mm_transitions))
        auto lifetime_mgr0  = std::make_shared<BlobLifetimeManager>();                           // Create lifetime manager
        auto lifetime_mgr1  = std::make_shared<BlobLifetimeManager>();                           // Create lifetime manager
        auto pool_mgr0      = std::make_shared<PoolManager>();                                   // Create pool manager
        auto pool_mgr1      = std::make_shared<PoolManager>();                                   // Create pool manager
        auto mm_layers      = std::make_shared<MemoryManagerOnDemand>(lifetime_mgr0, pool_mgr0); // Create the memory manager
        auto mm_transitions = std::make_shared<MemoryManagerOnDemand>(lifetime_mgr1, pool_mgr1); // Create the memory manager

        // The weights and biases tensors should be initialized with the values inferred with the training

        // Set memory manager where allowed to manage internal memory requirements
        // conv0   = arm_compute::support::cpp14::make_unique<NEWinogradConvolutionLayer>(mm_layers);
        conv0   = std::make_unique<NEWinogradConvolutionLayer>(mm_layers);

        // size_t ic = strtol(argv[1], nullptr, 10);
        // size_t oc = strtol(argv[2], nullptr, 10);
        // size_t ow = strtol(argv[3], nullptr, 10);

        /* [Initialize tensors] */

        // Initialize src tensor
        constexpr unsigned int width_src_image  = ow - 1 + 3;
        constexpr unsigned int height_src_image = ow - 1 + 3;
        constexpr unsigned int ifm_src_img      = ic;

        const TensorShape src_shape(width_src_image, height_src_image, ifm_src_img);
        src.allocator()->init(TensorInfo(src_shape, 1, DataType::F32));

        // Initialize tensors of conv0
        constexpr unsigned int kernel_x_conv0 = 3;
        constexpr unsigned int kernel_y_conv0 = 3;
        constexpr unsigned int ofm_conv0      = oc;

        const TensorShape weights_shape_conv0(kernel_x_conv0, kernel_y_conv0, src_shape.z(), ofm_conv0);
        const TensorShape biases_shape_conv0(weights_shape_conv0[3]);
        const TensorShape out_shape_conv0(src_shape.x(), src_shape.y(), weights_shape_conv0[3]);

        weights0.allocator()->init(TensorInfo(weights_shape_conv0, 1, DataType::F32));
        biases0.allocator()->init(TensorInfo(biases_shape_conv0, 1, DataType::F32));
        out_conv0.allocator()->init(TensorInfo(out_shape_conv0, 1, DataType::F32));

        // constexpr auto data_layout = DataLayout::NCHW;

        /* -----------------------End: [Initialize tensors] */

        /* [Configure functions] */

        // in:32x32x1: 3x3 convolution, 8 output features maps (OFM)
        conv0->configure(&src, &weights0, &biases0, &out_conv0, PadStrideInfo(1 /* stride_x */, 1 /* stride_y */, 1 /* pad_x */, 1 /* pad_y */));

        /* -----------------------End: [Configure functions] */

        /*[ Add tensors to memory manager ]*/

        // We call explicitly allocate after manage() in order to avoid overlapping lifetimes
        // memory_group0 = arm_compute::support::cpp14::make_unique<MemoryGroup>(mm_transitions);
        memory_group0 = std::make_unique<MemoryGroup>(mm_transitions);

        memory_group0->manage(&out_conv0);
        out_conv0.allocator()->allocate();

        /* -----------------------End: [ Add tensors to memory manager ] */

        /* [Allocate tensors] */

        // Now that the padding requirements are known we can allocate all tensors
        src.allocator()->allocate();
        weights0.allocator()->allocate();
        biases0.allocator()->allocate();

        /* -----------------------End: [Allocate tensors] */

        // Populate the layers manager. (Validity checks, memory allocations etc)
        mm_layers->populate(allocator, 1 /* num_pools */);

        // Populate the transitions manager. (Validity checks, memory allocations etc)
        mm_transitions->populate(allocator, 2 /* num_pools */);

        return true;
    }
    void do_run() override
    {
        // Acquire memory for the memory groups
        memory_group0->acquire();
        int n_loops = 100;
        int n_warmup = 5;

        for (int i=0; i<n_warmup; i++)
            conv0->run();

        Timer t;
        for (int i=0; i<n_loops; i++)
            conv0->run();
        float latency = t.getTime();
        // float gflops = M * N * K * 2 / latency * n_loops / 1000000;
        float gflops = 2.0 * oc * ic * 3 * 3 * ow * ow / latency * n_loops / 1000000.0;

        printf("ACL latency: %.6f ms\n", latency / n_loops);
	printf("GFlops: %.6f \n", gflops);

        // Release memory
        memory_group0->release();
    }

private:
    // The src tensor should contain the input image
    Tensor src{};

    // Intermediate tensors used
    Tensor weights0{};
    Tensor biases0{};
    Tensor out_conv0{};

    // NEON allocator
    Allocator allocator{};

    // Memory groups
    std::unique_ptr<MemoryGroup> memory_group0{};

    // Layers
    std::unique_ptr<NEWinogradConvolutionLayer>    conv0{};
};

/** Main program for cnn test
 *
 * The example implements the following CNN architecture:
 *
 * Input -> conv0:5x5 -> act0:relu -> pool:2x2 -> conv1:3x3 -> act1:relu -> pool:2x2 -> fc0 -> act2:relu -> softmax
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments
 */
int main(int argc, char **argv)
{
    int thread = atoi(argv[1]); 
    printf("thread number: %d\n", thread);
    arm_compute::Scheduler::get().set_num_threads(thread);
    return utils::run_example<NEONCNNExample>(argc, argv);
}
