# LoadImageToLocal

**页面ID:** atlasascendc_api_07_0241  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0241.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品AI Core | √ |
| Atlas 推理系列产品Vector Core | x |
| Atlas 训练系列产品 | x |

#### 功能说明

将图像数据从Global Memory搬运到Local Memory。 搬运过程中可以完成图像预处理操作：包括图像翻转，改变图像尺寸（抠图，裁边，缩放，伸展），以及色域转换，类型转换等。图像预处理的相关参数通过SetAippFunctions进行配置。

#### 函数原型

```
template <typename T>
__aicore__ inline void LoadImageToLocal(const LocalTensor<T>& dst, const LoadImageToLocalParams& loadDataParams)
```

#### 参数说明

**表1 **参数说明

| 参数名称 | 输入/输出 | 含义 |
| --- | --- | --- |
| dst | 输出 | 目的操作数，类型为LocalTensor。 LocalTensor的起始地址需要保证32字节对齐。 Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持数据类型：int8_t、half；支持的TPosition为A1、B1。 Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持数据类型：int8_t、half；支持的TPosition为A1、B1。 Atlas 200I/500 A2 推理产品 支持数据类型为： uint8_t、int8_t、half；支持的TPosition为A1、B1。 Atlas 推理系列产品AI Core，支持的数据类型为：uint8_t、int8_t、half；支持的TPosition为A1、B1。 |
| loadDataParams | 输入 | LoadData参数结构体，类型为LoadImageToLocalParams。 具体定义请参考${INSTALL_DIR}/include/ascendc/basic_api/interface/kernel_struct_mm.h，${INSTALL_DIR}请替换为CANN软件安装后文件存储路径。 参数说明参考表2。 |

**表2 **LoadImageToLocalParams结构体内参数说明

| 参数名称 | 输入/输出 | 含义 |
| --- | --- | --- |
| horizSize | 输入 | 从源图中加载图片的水平宽度，单位为像素，取值范围：horizSize∈[2, 4095] 。 |
| vertSize | 输入 | 从源图中加载图片的垂直高度，单位为像素，取值范围：vertSize∈[2, 4095]。 |
| horizStartPos | 输入 | 加载图片在源图片上的水平起始地址，单位为像素，取值范围：horizStartPos∈[0, 4095] 。默认为0。 注意：当输入图片为YUV420SP、XRGB8888， RGB888和YUV400格式时，该参数需要是偶数。 |
| vertStartPos | 输入 | 加载图片在源图片上的垂直起始地址，单位为像素，取值范围：vertStartPos∈[0, 4095] 。默认为0。 注意：当输入图片为YUV420SP格式时，该参数需要是偶数。 |
| srcHorizSize | 输入 | 源图像水平宽度 ，单位为像素，取值范围：srcHorizSize∈[2, 4095] 。 注意：当输入图片为YUV420SP格式时，该参数需要是偶数。 |
| topPadSize | 输入 | 目的图像顶部填充的像素数 ，取值范围：topPadSize∈[0, 32] ，默认为0。进行数据填充时使用，需要先调用SetAippFunctions通过AippPaddingParams配置填充的数值，再通过topPadSize、botPadSize、leftPadSize、rightPadSize配置填充的大小范围。 |
| botPadSize | 输入 | 目的图像底部填充的像素数，取值范围：botPadSize∈[0, 32] ，默认为0。 |
| leftPadSize | 输入 | 目的图像左边填充的像素数，取值范围：leftPadSize∈[0, 32] ，默认为0。 |
| rightPadSize | 输入 | 目的图像右边填充的像素数，取值范围：rightPadSize∈[0, 32] ，默认为0。 |
| sid | 输入 | 预留参数。为后续的功能做保留，开发者暂时无需关注，使用默认值即可。 |

#### 约束说明

- 加载到dst的图片的大小加padding的大小必须小于等于所在存储空间的大小。
- 当通过SetAippFunctions配置padding模式为块填充模式或者镜像块填充模式时，因为padding的数据来自于抠出的图片，左右padding的长度（leftPadSize、rightPadSize）必须小于或等于抠图的水平长度（horizSize），上下padding的长度（topPadSize、botPadSize）必须小于或等于抠图的垂直的长度（vertSize）。

#### 调用示例

该调用示例支持的运行平台为Atlas 推理系列产品AI Core，示例图片格式为YUV420SP。

```
#include "kernel_operator.h"

class KernelLoadImage {
public:
    __aicore__ inline KernelLoadImage()
    {
        // YUV420SP 图片中，Y 维度的 size
        gmSrc0Size = srcHorizSize * srcVertSize;
        // YUV420SP 图片中，UV 维度的 size
        gmSrc1Size = (srcHorizSize / 2) * (srcVertSize / 2) * 2;
        dstSize = dstHorizSize * dstVertSize * cSize;
    }
    __aicore__ inline void Init(__gm__ uint8_t *fmGm, __gm__ uint8_t *dstGm)
    {
        fmGlobal.SetGlobalBuffer((__gm__ uint8_t *)fmGm);
        dstGlobal.SetGlobalBuffer((__gm__ int8_t *)dstGm);
        pipe.InitBuffer(inQueueA1, 1, (gmSrc0Size + gmSrc1Size) * sizeof(int8_t));
        pipe.InitBuffer(outQueueUB, 1, dstSize * sizeof(int8_t));
    }
    __aicore__ inline void Process()
    {
        CopyIn();
        CopyToUB();
        CopyOut();
    }

private:
    __aicore__ inline void CopyIn()
    {
        AscendC::LocalTensor<int8_t> featureMapA1 = inQueueA1.AllocTensor<int8_t>();
        uint64_t fm_addr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(fmGlobal.GetPhyAddr()));
        // aipp config
        AscendC::AippParams<int8_t> aippConfig;
        aippConfig.cPaddingParams.cPaddingMode = cPadMode;
        aippConfig.cPaddingParams.cPaddingValue = cPaddingValue;
        // fmGlobal为整张输入图片，src1参数处填入图片UV维度的起始地址
        AscendC::SetAippFunctions(fmGlobal, fmGlobal[gmSrc0Size], inputFormat, aippConfig);
        AscendC::LoadImageToLocal(featureMapA1, { horizSize, vertSize, horizStartPos, vertStartPos, srcHorizSize, topPadSize, botPadSize, leftPadSize, rightPadSize });
        inQueueA1.EnQue(featureMapA1);
    }
    __aicore__ inline void CopyToUB()
    {
        AscendC::LocalTensor<int8_t> featureMapA1 = inQueueA1.DeQue<int8_t>();
        AscendC::LocalTensor<int8_t> featureMapUB = outQueueUB.AllocTensor<int8_t>();
        AscendC::DataCopy(featureMapUB, featureMapA1, dstSize);
        event_t eventIdMTE1ToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE1_MTE3));
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE3>(eventIdMTE1ToMTE3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE3>(eventIdMTE1ToMTE3);
        outQueueUB.EnQue<int8_t>(featureMapUB);
        inQueueA1.FreeTensor(featureMapA1);
    }
    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<int8_t> featureMapUB = outQueueUB.DeQue<int8_t>();
        AscendC::DataCopy(dstGlobal, featureMapUB, dstSize);
        outQueueUB.FreeTensor(featureMapUB);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::A1, 1> inQueueA1;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueUB;

    AscendC::GlobalTensor<uint8_t> fmGlobal;
    AscendC::GlobalTensor<int8_t> dstGlobal;

    uint16_t horizSize = 32, vertSize = 32, horizStartPos = 0, vertStartPos = 0, srcHorizSize = 32, srcVertSize = 32, leftPadSize = 0, rightPadSize = 0;
    uint32_t dstHorizSize = 32, dstVertSize = 32, cSize = 32;
    uint8_t topPadSize = 0, botPadSize = 0;
    uint32_t gmSrc0Size = 0, gmSrc1Size = 0, dstSize = 0;
    AscendC::AippInputFormat inputFormat = AscendC::AippInputFormat::YUV420SP_U8;
    uint32_t cPadMode = 0;
    int8_t cPaddingValue = 0;
};

extern "C" __global__ __aicore__ void load_image_simple_kernel(__gm__ uint8_t *fmGm, __gm__ uint8_t *dstGm)
{
    KernelLoadImage op;
    op.Init(fmGm, dstGm);
    op.Process();
}
```
