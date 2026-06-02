# InitConstValue

**页面ID:** atlasascendc_api_07_0237  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0237.html

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

将特定TPosition的LocalTensor初始化为某一具体数值。

#### 函数原型

```
template <typename T, typename U = PrimT<T>, typename Std::enable_if<Std::is_same<PrimT<T>, U>::value, bool>::type = true>
__aicore__ inline void InitConstValue(const LocalTensor<T>& dst, const InitConstValueParams<U>& initConstValueParams)
```

#### 参数说明

**表1 **模板参数说明

| 参数名 | 描述 |
| --- | --- |
| T | dst的数据类型。 Atlas 训练系列产品，支持的数据类型为：half Atlas 推理系列产品AI Core，支持的数据类型为：half/int16_t/uint16_t Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持的数据类型为：half/int16_t/uint16_t/bfloat16_t/float/int32_t/uint32_t Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持的数据类型为：half/int16_t/uint16_t/bfloat16_t/float/int32_t/uint32_t Atlas 200I/500 A2 推理产品，支持的数据类型为：half/int16_t/uint16_t/bfloat16_t/float/int32_t/uint32_t |
| U | 初始化值的数据类型。 - 当dst使用基础数据类型时， U和dst的数据类型T需保持一致，否则编译失败。- 当dst使用TensorTrait类型时，U和dst的数据类型T的LiteType需保持一致，否则编译失败。 最后一个模板参数仅用于上述数据类型检查，用户无需关注。 |

**表2 **参数说明

| 参数名称 | 输入/输出 | 含义 |
| --- | --- | --- |
| dst | 输出 | 目的操作数，结果矩阵，类型为LocalTensor。 Atlas 训练系列产品，支持的TPosition为A1/A2/B1/B2。 Atlas 推理系列产品AI Core，支持的TPosition为A1/A2/B1/B2。 Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持的TPosition为A1/A2/B1/B2。 Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持的TPosition为A1/A2/B1/B2。 Atlas 200I/500 A2 推理产品，支持的TPosition为A1/A2/B1/B2。 如果TPosition为A1/B1，起始地址需要满足32B对齐；如果TPosition为A2/B2，起始地址需要满足512B对齐。 |
| InitConstValueParams | 输入 | 初始化相关参数，类型为InitConstValueParams。 具体定义请参考${INSTALL_DIR}/include/ascendc/basic_api/interface/kernel_struct_mm.h，${INSTALL_DIR}请替换为CANN软件安装后文件存储路径。 参数说明请参考表3。 Atlas 训练系列产品，仅支持配置迭代次数（repeatTimes）和初始化值（initValue）。 Atlas 推理系列产品AI Core，仅支持配置迭代次数（repeatTimes）和初始化值（initValue）。 Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持配置所有参数。 Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持配置所有参数。 Atlas 200I/500 A2 推理产品，支持配置所有参数。 - 仅支持配置迭代次数（repeatTimes）和初始化值（initValue）场景下，其他参数配置无效。每次迭代处理固定数据量（512字节），迭代间无间隔。- 支持配置所有参数场景下，支持配置迭代次数（repeatTimes）、初始化值（initValue）、每个迭代处理的数据块个数（blockNum）和迭代间间隔（dstGap）。 |

**表3 **InitConstValueParams结构体参数说明

| 参数名称 | 含义 |
| --- | --- |
| repeatTimes | 迭代次数。默认值为0。 - 仅支持配置迭代次数（repeatTimes）和初始化值（initValue）场景下，repeatTimes∈[0, 255]。- 支持配置所有参数场景下，repeatTimes∈[0, 32767] 。 |
| blockNum | 每次迭代初始化的数据块个数，取值范围：blockNum∈[0, 32767] 。默认值为0。 - dst的位置为A1/B1时，每一个block（数据块）大小是32B；- dst的位置为A2/B2时，每一个block（数据块）大小是512B。 |
| dstGap | 目的操作数前一个迭代结束地址到后一个迭代起始地址之间的距离。 - dst的位置为A1/B1时，单位是32B；- dst的位置为A2/B2时，单位是512B。 取值范围：dstGap∈[0, 32767] 。默认值为0。 |
| initValue | 初始化的value值，支持的数据类型与dst保持一致。 |

#### 约束说明

#### 调用示例

```
#include "kernel_operator.h"

template <typename dst_T, typename fmap_T, typename weight_T, typename dstCO1_T> class KernelCubeMmad {
public:
    __aicore__ inline KernelCubeMmad()
    {
        C0 = 32 / sizeof(fmap_T);
        C1 = channelSize / C0;
        coutBlocks = (Cout + 16 - 1) / 16;
        ho = H - dilationH * (Kh - 1);
        wo = W - dilationW * (Kw - 1);
        howo = ho * wo;
        howoRound = ((howo + 16 - 1) / 16) * 16;
        featureMapA1Size = C1 * H * W * C0;      // shape: [C1, H, W, C0]
        weightA1Size = C1 * Kh * Kw * Cout * C0; // shape: [C1, Kh, Kw, Cout, C0]
        featureMapA2Size = howoRound * (C1 * Kh * Kw * C0);
        weightB2Size = (C1 * Kh * Kw * C0) * coutBlocks * 16;
        m = howo;
        k = C1 * Kh * Kw * C0;
        n = Cout;
        biasSize = Cout;                  // shape: [Cout]
        dstSize = coutBlocks * howo * 16; // shape: [coutBlocks, howo, 16]
        dstCO1Size = coutBlocks * howoRound * 16;
        fmRepeat = featureMapA2Size / (16 * C0);
        weRepeat = weightB2Size / (16 * C0);
    }
    __aicore__ inline void Init(__gm__ uint8_t* fmGm, __gm__ uint8_t* weGm, __gm__ uint8_t* biasGm,
        __gm__ uint8_t* dstGm)
    {
        fmGlobal.SetGlobalBuffer((__gm__ fmap_T*)fmGm);
        weGlobal.SetGlobalBuffer((__gm__ weight_T*)weGm);
        biasGlobal.SetGlobalBuffer((__gm__ dstCO1_T*)biasGm);
        dstGlobal.SetGlobalBuffer((__gm__ dst_T*)dstGm);
        pipe.InitBuffer(inQueueFmA1, 1, featureMapA1Size * sizeof(fmap_T));
        pipe.InitBuffer(inQueueFmA2, 1, featureMapA2Size * sizeof(fmap_T));
        pipe.InitBuffer(inQueueWeB1, 1, weightA1Size * sizeof(weight_T));
        pipe.InitBuffer(inQueueWeB2, 1, weightB2Size * sizeof(weight_T));
        pipe.InitBuffer(inQueueBiasA1, 1, biasSize * sizeof(dstCO1_T));
        pipe.InitBuffer(outQueueCO1, 1, dstCO1Size * sizeof(dstCO1_T));
    }
    __aicore__ inline void Process()
    {
        CopyIn();
        Split();
        Compute();
        CopyOut();
    }

private:
    __aicore__ inline void CopyIn()
    {
        AscendC::LocalTensor<fmap_T> featureMapA1 = inQueueFmA1.AllocTensor<fmap_T>();
        AscendC::LocalTensor<weight_T> weightB1 = inQueueWeB1.AllocTensor<weight_T>();
        AscendC::LocalTensor<dstCO1_T> biasA1 = inQueueBiasA1.AllocTensor<dstCO1_T>();

        AscendC::InitConstValue(featureMapA1, {1, static_cast<uint16_t>(featureMapA1Size * sizeof(fmap_T) / 32), 0, 1});
        AscendC::InitConstValue(weightB1, {1, static_cast<uint16_t>(weightA1Size * sizeof(weight_T) / 32), 0, 2});
        AscendC::DataCopy(biasA1, biasGlobal, { 1, static_cast<uint16_t>(biasSize * sizeof(dstCO1_T) / 32), 0, 0 });

        inQueueFmA1.EnQue(featureMapA1);
        inQueueWeB1.EnQue(weightB1);
        inQueueBiasA1.EnQue(biasA1);
    }
    __aicore__ inline void Split()
    {
        AscendC::LocalTensor<fmap_T> featureMapA1 = inQueueFmA1.DeQue<fmap_T>();
        AscendC::LocalTensor<weight_T> weightB1 = inQueueWeB1.DeQue<weight_T>();
        AscendC::LocalTensor<fmap_T> featureMapA2 = inQueueFmA2.AllocTensor<fmap_T>();
        AscendC::LocalTensor<weight_T> weightB2 = inQueueWeB2.AllocTensor<weight_T>();

        AscendC::InitConstValue(featureMapA2, {1, static_cast<uint16_t>(featureMapA2Size * sizeof(fmap_T) / 512), 0, 1});
        AscendC::InitConstValue(weightB2, { 1, static_cast<uint16_t>(weightB2Size * sizeof(weight_T) / 512), 0, 2});

        inQueueFmA2.EnQue<fmap_T>(featureMapA2);
        inQueueWeB2.EnQue<weight_T>(weightB2);
        inQueueFmA1.FreeTensor(featureMapA1);
        inQueueWeB1.FreeTensor(weightB1);
    }
    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<fmap_T> featureMapA2 = inQueueFmA2.DeQue<fmap_T>();
        AscendC::LocalTensor<weight_T> weightB2 = inQueueWeB2.DeQue<weight_T>();
        AscendC::LocalTensor<dstCO1_T> dstCO1 = outQueueCO1.AllocTensor<dstCO1_T>();
        AscendC::LocalTensor<dstCO1_T> biasA1 = inQueueBiasA1.DeQue<dstCO1_T>();
        AscendC::Mmad(dstCO1, featureMapA2, weightB2, biasA1, { m, n, k, true, 0, false, false, false });

        outQueueCO1.EnQue<dstCO1_T>(dstCO1);
        inQueueFmA2.FreeTensor(featureMapA2);
        inQueueWeB2.FreeTensor(weightB2);
        inQueueBiasA1.FreeTensor(biasA1);
    }
    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<dstCO1_T> dstCO1 = outQueueCO1.DeQue<dstCO1_T>();
        AscendC::FixpipeParamsV220 fixpipeParams;
        fixpipeParams.nSize = coutBlocks * 16;
        fixpipeParams.mSize = howo;
        fixpipeParams.srcStride = howo;
        fixpipeParams.dstStride = howo * AscendC::BLOCK_CUBE * sizeof(dst_T) / AscendC::ONE_BLK_SIZE;
        fixpipeParams.quantPre = deqMode;
        AscendC::Fixpipe<dst_T, dstCO1_T, AscendC::CFG_NZ>(dstGlobal, dstCO1, fixpipeParams);
        outQueueCO1.FreeTensor(dstCO1);
    }

private:
    AscendC::TPipe pipe;
    // feature map queue
    AscendC::TQue<AscendC::TPosition::A1, 1> inQueueFmA1;
    AscendC::TQue<AscendC::TPosition::A2, 1> inQueueFmA2;
    // weight queue
    AscendC::TQue<AscendC::TPosition::B1, 1> inQueueWeB1;
    AscendC::TQue<AscendC::TPosition::B2, 1> inQueueWeB2;
    // bias queue
    AscendC::TQue<AscendC::TPosition::A1, 1> inQueueBiasA1;
    // dst queue
    AscendC::TQue<AscendC::TPosition::CO1, 1> outQueueCO1;

    AscendC::GlobalTensor<fmap_T> fmGlobal;
    AscendC::GlobalTensor<weight_T> weGlobal;
    AscendC::GlobalTensor<dst_T> dstGlobal;
    AscendC::GlobalTensor<dstCO1_T> biasGlobal;

    uint16_t channelSize = 32;
    uint16_t H = 4, W = 4;
    uint8_t Kh = 2, Kw = 2;
    uint16_t Cout = 16;
    uint16_t C0, C1;
    uint8_t dilationH = 2, dilationW = 2;
    uint16_t coutBlocks, ho, wo, howo, howoRound;
    uint32_t featureMapA1Size, weightA1Size, featureMapA2Size, weightB2Size, biasSize, dstSize, dstCO1Size;
    uint16_t m, k, n;
    uint8_t fmRepeat, weRepeat;
    AscendC::QuantMode_t deqMode = AscendC::QuantMode_t::F322F16;
};

extern "C" __global__ __aicore__ void cube_mmad_simple_kernel(__gm__ uint8_t *fmGm, __gm__ uint8_t *weGm,
    __gm__ uint8_t *biasGm, __gm__ uint8_t *dstGm)
{
    KernelCubeMmad<half, half, half, half> op;
    op.Init(fmGm, weGm, biasGm, dstGm);
    op.Process();
}
```
