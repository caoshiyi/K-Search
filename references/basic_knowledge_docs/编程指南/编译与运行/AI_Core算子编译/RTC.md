# RTC

**页面ID:** atlas_ascendc_10_00040  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_10_00040.html

---

RTC是Ascend C运行时编译库，通过aclrtc接口，在程序运行时，将中间代码动态编译成目标机器码，提升程序运行性能。

运行时编译库提供以下核心接口：

- aclrtcCreateProg：根据输入参数（字符串形式表达的Ascend C源代码等）创建aclrtcProg程序实例。
- aclrtcCompileProg：编译给定的程序，支持用户自定义编译选项，比如指定NPU架构版本号：--npu-arch=dav-2201。支持的编译选项可以参考毕昇编译器编译选项。
- aclrtcGetBinDataSize：获取编译后的Device侧二进制数据的大小。
- aclrtcGetBinData：获取编译后的Device侧二进制数据。
- aclrtcDestroyProg：在编译和执行过程结束后，销毁给定的程序。

编译完成后需要调用如下接口完成（仅列出核心接口）Kernel加载与执行。完整流程和详细接口说明请参考“Kernel加载与执行”章节。

1. 通过aclrtBinaryLoadFromData接口解析由aclrtcGetBinData接口获取的算子二进制数据。
2. 获取核函数句柄并根据核函数句柄操作其参数列表，相关接口包括aclrtBinaryGetFunction（获取核函数句柄）、aclrtKernelArgsInit（初始化参数列表）、aclrtKernelArgsAppend（追加拷贝用户设置的参数值如xDevice, yDevice, zDevice）等。
3. 调用aclrtLaunchKernelWithConfig接口，启动对应算子的计算任务。

    如下是一个使用aclrtc接口编译并运行Add自定义算子的完整样例：

```
#include <iostream>
#include <fstream>
#include <vector>
#include "acl/acl.h"
// 使用aclrtc接口需要包含的头文件
#include "acl/acl_rt_compile.h"

#define CHECK_ACL(x)                                                                        \
    do {                                                                                    \
        aclError __ret = x;                                                                 \
        if (__ret != ACL_ERROR_NONE) {                                                      \
            std::cerr << __FILE__ << ":" << __LINE__ << " aclError:" << __ret << std::endl; \
        }                                                                                   \
    } while (0);

int main(int argc, char *argv[])
{
    // ----------------------------------------------------- aclrtc part -----------------------------------------------------
    const char *src = R""""(
#include "kernel_operator.h"
constexpr int32_t TOTAL_LENGTH = 8 * 1024;                            // total length of data
constexpr int32_t USE_CORE_NUM = 8;                                   // num of core used
constexpr int32_t BLOCK_LENGTH = TOTAL_LENGTH / USE_CORE_NUM;         // length computed of each core
constexpr int32_t TILE_NUM = 8;                                       // split data into 8 tiles for each core
constexpr int32_t BUFFER_NUM = 2;                                     // tensor num for each queue
constexpr int32_t TILE_LENGTH = BLOCK_LENGTH / TILE_NUM / BUFFER_NUM; // separate to 2 parts, due to double buffer

class KernelAdd {
public:
    __aicore__ inline KernelAdd() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z)
    {
        xGm.SetGlobalBuffer((__gm__ float *)x + BLOCK_LENGTH * AscendC::GetBlockIdx(), BLOCK_LENGTH);
        yGm.SetGlobalBuffer((__gm__ float *)y + BLOCK_LENGTH * AscendC::GetBlockIdx(), BLOCK_LENGTH);
        zGm.SetGlobalBuffer((__gm__ float *)z + BLOCK_LENGTH * AscendC::GetBlockIdx(), BLOCK_LENGTH);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, TILE_LENGTH * sizeof(float));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, TILE_LENGTH * sizeof(float));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, TILE_LENGTH * sizeof(float));
    }
    __aicore__ inline void Process()
    {
        int32_t loopCount = TILE_NUM * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        AscendC::LocalTensor<float> yLocal = inQueueY.AllocTensor<float>();
        AscendC::DataCopy(xLocal, xGm[progress * TILE_LENGTH], TILE_LENGTH);
        AscendC::DataCopy(yLocal, yGm[progress * TILE_LENGTH], TILE_LENGTH);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> yLocal = inQueueY.DeQue<float>();
        AscendC::LocalTensor<float> zLocal = outQueueZ.AllocTensor<float>();
        AscendC::Add(zLocal, xLocal, yLocal, TILE_LENGTH);

        outQueueZ.EnQue<float>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<float> zLocal = outQueueZ.DeQue<float>();
        AscendC::DataCopy(zGm[progress * TILE_LENGTH], zLocal, TILE_LENGTH);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    AscendC::GlobalTensor<float> zGm;
};
extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    KernelAdd op;
    op.Init(x, y, z);
    op.Process();
}
)"""";
    // aclrtc流程，src为用户Device侧源码，通过aclrtcCreateProg来创建编译程序
    aclrtcProg prog;
    CHECK_ACL(aclrtcCreateProg(&prog, src, "add_custom", 0, nullptr, nullptr));

    // aclrtc流程，传入毕昇编译器的编译选项，调用aclrtcCompileProg进行编译
    const char *options[] = {
        "--npu-arch=dav-2201",
    };
    int numOptions = sizeof(options) / sizeof(options[0]);
    CHECK_ACL(aclrtcCompileProg(prog, numOptions, options));

    // aclrtc流程，获取Device侧二进制内容和大小
    size_t binDataSizeRet;
    CHECK_ACL(aclrtcGetBinDataSize(prog, &binDataSizeRet));
    std::vector<char> deviceELF(binDataSizeRet);
    CHECK_ACL(aclrtcGetBinData(prog, deviceELF.data()));

    const char *funcName = "add_custom";
    // ----------------------------------------------------- aclrt part -----------------------------------------------------
    uint32_t blockDim = 8;
    size_t inputByteSize = 8 * 1024 * sizeof(uint32_t);
    size_t outputByteSize = 8 * 1024 * sizeof(uint32_t);
    CHECK_ACL(aclInit(nullptr));
    int32_t deviceId = 0;
    CHECK_ACL(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    uint8_t *xHost, *yHost, *zHost;
    uint8_t *xDevice, *yDevice, *zDevice;
    CHECK_ACL(aclrtMallocHost((void **)(&xHost), inputByteSize));
    CHECK_ACL(aclrtMallocHost((void **)(&yHost), inputByteSize));
    CHECK_ACL(aclrtMallocHost((void **)(&zHost), outputByteSize));
    CHECK_ACL(aclrtMalloc((void **)&xDevice, inputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&yDevice, inputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&zDevice, outputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemcpy(xDevice, inputByteSize, xHost, inputByteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(yDevice, inputByteSize, yHost, inputByteSize, ACL_MEMCPY_HOST_TO_DEVICE));

    aclrtBinHandle binHandle = nullptr;
    aclrtBinaryLoadOptions loadOption;
    loadOption.numOpt = 1;
    aclrtBinaryLoadOption option;
    option.type = ACL_RT_BINARY_LOAD_OPT_LAZY_MAGIC;
    option.value.magic = ACL_RT_BINARY_MAGIC_ELF_VECTOR_CORE;   // 设置magic值，表示算子在Vector Core上执行
    loadOption.options = &option;
    CHECK_ACL(aclrtBinaryLoadFromData(deviceELF.data(), binDataSizeRet, &loadOption, &binHandle));
    aclrtFuncHandle funcHandle = nullptr;

    CHECK_ACL(aclrtBinaryGetFunction(binHandle, funcName, &funcHandle));

    aclrtArgsHandle argsHandle = nullptr;
    aclrtParamHandle paramHandle = nullptr;
    CHECK_ACL(aclrtKernelArgsInit(funcHandle, &argsHandle));
    CHECK_ACL(aclrtKernelArgsAppend(argsHandle, (void **)&xDevice, sizeof(uintptr_t), &paramHandle));
    CHECK_ACL(aclrtKernelArgsAppend(argsHandle, (void **)&yDevice, sizeof(uintptr_t), &paramHandle));
    CHECK_ACL(aclrtKernelArgsAppend(argsHandle, (void **)&zDevice, sizeof(uintptr_t), &paramHandle));
    CHECK_ACL(aclrtKernelArgsFinalize(argsHandle));
    // 核函数入口
    CHECK_ACL(aclrtLaunchKernelWithConfig(funcHandle, blockDim, stream, nullptr, argsHandle, nullptr));

    CHECK_ACL(aclrtSynchronizeStream(stream));
    CHECK_ACL(aclrtMemcpy(zHost, outputByteSize, zDevice, outputByteSize, ACL_MEMCPY_DEVICE_TO_HOST));

    // 获取日志大小并得到日志字符串
    size_t logSize;
    CHECK_ACL(aclrtcGetCompileLogSize(prog, &logSize));
    char* log = (char*)malloc(logSize);
    CHECK_ACL(aclrtcGetCompileLog(prog, log));
    // 将日志字符串存到文件中
    /*
    std::ofstream logFile("compile.log");
    if (logFile.is_open()) {
	logFile << log << std::endl;
	logFile.close();
	std::cout << "already write to compile.log!" << std::endl;
    }
    */

    CHECK_ACL(aclrtBinaryUnLoad(binHandle));
    CHECK_ACL(aclrtFree(xDevice));
    CHECK_ACL(aclrtFree(yDevice));
    CHECK_ACL(aclrtFree(zDevice));
    CHECK_ACL(aclrtFreeHost(xHost));
    CHECK_ACL(aclrtFreeHost(yHost));
    CHECK_ACL(aclrtFreeHost(zHost));
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());

    // 编译和运行均已结束，销毁程序
    CHECK_ACL(aclrtcDestroyProg(&prog));

    return 0;
}
```

编译命令如下，编译时需要设置-I路径为${INSTALL_DIR}/include，用于找到aclrtc相关头文件，并需要链接alc_rtc动态库。

```
g++ add_custom.cpp -I${INSTALL_DIR}/include -L${INSTALL_DIR}/lib64 -lascendcl -lacl_rtc -o main
```

${INSTALL_DIR}请替换为CANN软件安装后文件存储路径。以root用户安装为例，则安装后文件存储路径为：/usr/local/Ascend/cann。
