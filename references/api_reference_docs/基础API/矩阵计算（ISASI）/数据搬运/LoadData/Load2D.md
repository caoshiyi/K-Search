# Load2D

**页面ID:** atlasascendc_api_07_00169  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_00169.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品AI Core | √ |
| Atlas 推理系列产品Vector Core | x |
| Atlas 训练系列产品 | √ |

#### 功能说明

Load2D支持如下数据通路的搬运：

GM->A1; GM->B1; GM->A2; GM->B2;

A1->A2; B1->B2。

#### 函数原型

- Load2D接口

```
template <typename T>
__aicore__ inline void LoadData(const LocalTensor<T>& dst, const LocalTensor<T>& src, const LoadData2DParams& loadDataParams)
template <typename T> 
__aicore__ inline void LoadData(const LocalTensor<T>& dst, const GlobalTensor<T>& src, const LoadData2DParams& loadDataParams)
```

#### 参数说明

**表1 **模板参数说明

| 参数名称 | 含义 |
| --- | --- |
| T | 源操作数和目的操作数的数据类型。 - **Load2D接口**Atlas 训练系列产品，支持的数据类型为：uint8_t/int8_t/uint16_t/int16_t/half Atlas 推理系列产品AI Core，支持的数据类型为：uint8_t/int8_t/uint16_t/int16_t/half Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持数据类型为：uint8_t/int8_t/uint16_t/int16_t/half/bfloat16_t/uint32_t/int32_t/float Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持数据类型为：uint8_t/int8_t/uint16_t/int16_t/half/bfloat16_t/uint32_t/int32_t/float Atlas 200I/500 A2 推理产品，支持数据类型为：uint8_t/int8_t/uint16_t/int16_t/half/bfloat16_t/uint32_t/int32_t/float |

**表2 **通用参数说明

| 参数名称 | 输入/输出 | 含义 |
| --- | --- | --- |
| dst | 输出 | 目的操作数，类型为LocalTensor。 数据连续排列顺序由目的操作数所在TPosition决定，具体约束如下： - A2：ZZ格式；对应的分形大小为16 * (32B / sizeof(T))。- B2：ZN格式；对应的分形大小为 (32B / sizeof(T))  * 16。- A1/B1：无格式要求，一般情况下为NZ格式。NZ格式下，对应的分形大小为16 * (32B / sizeof(T))。 |
| src | 输入 | 源操作数，类型为LocalTensor或GlobalTensor。 数据类型需要与dst保持一致。 |
| loadDataParams | 输入 | LoadData参数结构体，类型为： - LoadData2DParams，具体参考表3。 上述结构体参数定义请参考${INSTALL_DIR}/include/ascendc/basic_api/interface/kernel_struct_mm.h，${INSTALL_DIR}请替换为CANN软件安装后文件存储路径。 |

**表3 **LoadData2DParams结构体内参数说明

| 参数名称 | 含义 |
| --- | --- |
| startIndex | 分形矩阵ID，说明搬运起始位置为源操作数中第几个分形（0为源操作数中第1个分形矩阵）。取值范围：startIndex∈[0, 65535] 。单位：512B。默认为0。 |
| repeatTimes | 迭代次数，每个迭代可以处理512B数据。取值范围：repeatTimes∈[1, 255]。 |
| srcStride | 相邻迭代间，源操作数前一个分形与后一个分形起始地址的间隔，单位：512B。取值范围：src_stride∈[0, 65535]。默认为0。 |
| sid | 预留参数，配置为0即可。 |
| dstGap | 相邻迭代间，目的操作数前一个分形结束地址与后一个分形起始地址的间隔，单位：512B。取值范围：dstGap∈[0, 65535]。默认为0。 注：Atlas 训练系列产品此参数不使能。 |
| ifTranspose | 是否启用转置功能，对每个分形矩阵进行转置，默认为false: - true：启用- false：不启用 注意：只有A1->A2和B1->B2通路才能使能转置，使能转置功能时，源操作数、目的操作数仅支持uint16_t/int16_t/half数据类型。 |
| addrMode | 预留参数，配置为0即可。 |

#### 约束说明

#### 调用示例

该调用示例支持的运行平台为Atlas 推理系列产品AI Core。

```
#include "kernel_operator.h"

class KernelLoadData {
public:
    __aicore__ inline KernelLoadData()
    {
        coutBlocks = (Cout + 16 - 1) / 16;
        ho = (H + padTop + padBottom - dilationH * (Kh - 1) - 1) / strideH + 1;
        wo = (W + padLeft + padRight - dilationW * (Kw - 1) - 1) / strideW + 1;
        howo = ho * wo;
        howoRound = ((howo + 16 - 1) / 16) * 16;
        featureMapA1Size = C1 * H * W * C0;      // shape: [C1, H, W, C0]
        weightA1Size = C1 * Kh * Kw * Cout * C0; // shape: [C1, Kh, Kw, Cout, C0]
        featureMapA2Size = howoRound * (C1 * Kh * Kw * C0);
        weightB2Size = (C1 * Kh * Kw * C0) * coutBlocks * 16;
        m = howo;
        k = C1 * Kh * Kw * C0;
        n = Cout;
        dstSize = coutBlocks * howo * 16; // shape: [coutBlocks, howo, 16]
        dstCO1Size = coutBlocks * howoRound * 16;
        fmRepeat = featureMapA2Size / (16 * C0);
        weRepeat = weightB2Size / (16 * C0);
    }
    __aicore__ inline void Init(__gm__ uint8_t* fmGm, __gm__ uint8_t* weGm, __gm__ uint8_t* dstGm)
    {
        fmGlobal.SetGlobalBuffer((__gm__ half*)fmGm);
        weGlobal.SetGlobalBuffer((__gm__ half*)weGm);
        dstGlobal.SetGlobalBuffer((__gm__ half*)dstGm);
        pipe.InitBuffer(inQueueFmA1, 1, featureMapA1Size * sizeof(half));
        pipe.InitBuffer(inQueueFmA2, 1, featureMapA2Size * sizeof(half));
        pipe.InitBuffer(inQueueWeB1, 1, weightA1Size * sizeof(half));
        pipe.InitBuffer(inQueueWeB2, 1, weightB2Size * sizeof(half));
        pipe.InitBuffer(outQueue
, 1, dstCO1Size * sizeof(float));
        pipe.InitBuffer(outQueueUB, 1, dstSize * sizeof(half));
    }
    __aicore__ inline void Process()
    {
        CopyIn();
        Split();
        Compute();
        CopyUB();
        CopyOut();
    }

private:
    __aicore__ inline void CopyIn()
    {
        AscendC::LocalTensor<half> featureMapA1 = inQueueFmA1.AllocTensor<half>();
        AscendC::LocalTensor<half> weightB1 = inQueueWeB1.AllocTensor<half>();
        AscendC::DataCopy(featureMapA1, fmGlobal, { 1, static_cast<uint16_t>(featureMapA1Size * sizeof(half) / 32), 0, 0 });
        AscendC::DataCopy(weightB1, weGlobal, { 1, static_cast<uint16_t>(weightA1Size * sizeof(half) / 32), 0, 0 });
        inQueueFmA1.EnQue(featureMapA1);
        inQueueWeB1.EnQue(weightB1);
    }
    __aicore__ inline void Split()
    {
        AscendC::LocalTensor<half> featureMapA1 = inQueueFmA1.DeQue<half>();
        AscendC::LocalTensor<half> weightB1 = inQueueWeB1.DeQue<half>();
        AscendC::LocalTensor<half> featureMapA2 = inQueueFmA2.AllocTensor<half>();
        AscendC::LocalTensor<half> weightB2 = inQueueWeB2.AllocTensor<half>();
        uint8_t padList[4] = {padLeft, padRight, padTop, padBottom};
        AscendC::LoadData(featureMapA2, featureMapA1,
            { padList, H, W, 0, 0, 0, -1, -1, strideW, strideH, Kw, Kh, dilationW, dilationH, 1, 0, fmRepeat, 0, (half)(0)});
        AscendC::LoadData(weightB2, weightB1, { 0, weRepeat, 1, 0, 0, false, 0 });
        inQueueFmA2.EnQue<half>(featureMapA2);
        inQueueWeB2.EnQue<half>(weightB2);
        inQueueFmA1.FreeTensor(featureMapA1);
        inQueueWeB1.FreeTensor(weightB1);
    }
    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<half> featureMapA2 = inQueueFmA2.DeQue<half>();
        AscendC::LocalTensor<half> weightB2 = inQueueWeB2.DeQue<half>();
        AscendC::LocalTensor<float> dstCO1 = outQueueCO1.AllocTensor<float>();
        AscendC::Mmad(dstCO1, featureMapA2, weightB2, { m, n, k, 0, false, true });
        outQueueCO1.EnQue<float>(dstCO1);
        inQueueFmA2.FreeTensor(featureMapA2);
        inQueueWeB2.FreeTensor(weightB2);
    }
    __aicore__ inline void CopyUB()
    {
        AscendC::LocalTensor<float> dstCO1 = outQueueCO1.DeQue<float>();
        AscendC::LocalTensor<half> dstUB = outQueueUB.AllocTensor<half>();
        AscendC::DataCopyParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = m * n * sizeof(float) / 1024;
        AscendC::DataCopyEnhancedParams enhancedParams;
        enhancedParams.blockMode = AscendC::BlockMode::BLOCK_MODE_MATRIX;
        AscendC::DataCopy(dstUB, dstCO1, dataCopyParams, enhancedParams);
        outQueueUB.EnQue<half>(dstUB);
        outQueueCO1.FreeTensor(dstCO1);
    }
    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<half> dstUB = outQueueUB.DeQue<half>();
        AscendC::DataCopy(dstGlobal, dstUB, m * n);
        outQueueUB.FreeTensor(dstUB);
    }

private:
    AscendC::TPipe pipe;
    // feature map queue
    AscendC::TQue<AscendC::TPosition::A1, 1> inQueueFmA1;
    AscendC::TQue<AscendC::TPosition::A2, 1> inQueueFmA2;
    // weight queue
    AscendC::TQue<AscendC::TPosition::B1, 1> inQueueWeB1;
    AscendC::TQue<AscendC::TPosition::B2, 1> inQueueWeB2;
    // dst queue
    AscendC::TQue<AscendC::TPosition::CO1, 1> outQueueCO1;
    AscendC::TQue<AscendC::TPosition::CO2, 1> outQueueUB;
    AscendC::GlobalTensor<half> fmGlobal, weGlobal, dstGlobal;
    uint16_t C1 = 2;
    uint16_t H = 4, W = 4;
    uint8_t Kh = 2, Kw = 2;
    uint16_t Cout = 16;
    uint16_t C0 = 16;
    uint8_t dilationH = 2, dilationW = 2;
    uint8_t padTop = 1, padBottom = 1, padLeft = 1, padRight = 1;
    uint8_t strideH = 1, strideW = 1;
    uint16_t coutBlocks, ho, wo, howo, howoRound;
    uint32_t featureMapA1Size, weightA1Size, featureMapA2Size, weightB2Size, dstSize, dstCO1Size;
    uint16_t m, k, n;
    uint8_t fmRepeat, weRepeat;
};

extern "C" __global__ __aicore__ void load_data_simple_kernel(__gm__ uint8_t* fmGm, __gm__ uint8_t* weGm,
    __gm__ uint8_t* dstGm)
{
    KernelLoadData op;
    op.Init(fmGm, weGm, dstGm);
    op.Process();
}
```
