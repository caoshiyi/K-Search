# SetFmatrix

**页面ID:** atlasascendc_api_07_0245  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0245.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | x |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | x |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品AI Core | √ |
| Atlas 推理系列产品Vector Core | x |
| Atlas 训练系列产品 | x |

#### 功能说明

用于调用Load3Dv1/Load3Dv2时设置FeatureMap的属性描述。Load3Dv1/Load3Dv2的模板参数isSetFMatrix设置为false时，表示Load3Dv1/Load3Dv2传入的FeatureMap的属性（包括l1H、l1W、padList，参数介绍参考表4 LoadData3DParamsV1结构体内参数说明、表5 LoadData3DParamsV2结构体内参数说明）描述不生效，开发者需要通过该接口进行设置。

#### 函数原型

```
__aicore__ inline void SetFmatrix(uint16_t l1H, uint16_t l1W, const uint8_t padList[4], const FmatrixMode& fmatrixMode)
```

#### 参数说明

**表1 **参数说明

| 参数名称 | 输入/输出 | 含义 |
| --- | --- | --- |
| l1H | 输入 | 源操作数height，取值范围：l1H∈[1, 32767]。 |
| l1W | 输入 | 源操作数width，取值范围：l1W∈[1, 32767] 。 |
| padList | 输入 | padding列表 [padding_left, padding_right, padding_top, padding_bottom]，每个元素取值范围：[0,255]。默认为{0, 0, 0, 0}。 |
| 用于控制LoadData指令从left还是right寄存器获取信息。FmatrixMode类型，定义如下。当前只支持FMATRIX_LEFT，左右矩阵均使用该配置。 ``` enum class FmatrixMode : uint8_t {     FMATRIX_LEFT = 0,     FMATRIX_RIGHT = 1, }; ``` |  |  |

#### 约束说明

- 该接口需要配合load3Dv1/load3Dv2接口一起使用，需要在load3Dv1/load3Dv2接口之前调用。

#### 调用示例

```
#include "kernel_operator.h"

template <typename dst_T, typename fmap_T, typename weight_T, typename dstCO1_T> class KernelLoad3d {
public:
    __aicore__ inline KernelLoad3d()
    {
        // ceiling of 16
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

        AscendC::DataCopy(featureMapA1, fmGlobal, { 1, static_cast<uint16_t>(featureMapA1Size * sizeof(fmap_T) / 32), 0, 0 });
        AscendC::DataCopy(weightB1, weGlobal, { 1, static_cast<uint16_t>(weightA1Size * sizeof(weight_T) / 32), 0, 0 });

        inQueueFmA1.EnQue(featureMapA1);
        inQueueWeB1.EnQue(weightB1);
    }
    __aicore__ inline void Split()
    {    
        AscendC::LocalTensor<fmap_T> featureMapA1 = inQueueFmA1.DeQue<fmap_T>();
        AscendC::LocalTensor<weight_T> weightB1 = inQueueWeB1.DeQue<weight_T>();
        AscendC::LocalTensor<fmap_T> featureMapA2 = inQueueFmA2.AllocTensor<fmap_T>();
        AscendC::LocalTensor<weight_T> weightB2 = inQueueWeB2.AllocTensor<weight_T>();

        uint8_t padList[PAD_SIZE] = {0, 0, 0, 0};
        AscendC::SetFmatrix(H, W, padList, FmatrixMode::FMATRIX_LEFT);
        AscendC::SetLoadDataPaddingValue(0);
        AscendC::SetLoadDataRepeat({0, 1, 0});
        AscendC::SetLoadDataBoundary((uint32_t)0);
        static constexpr AscendC::IsResetLoad3dConfig LOAD3D_CONFIG = {false,false};
        AscendC::LoadData<fmap_T, LOAD3D_CONFIG>(featureMapA2, featureMapA1,
            { padList, H, W, channelSize, k, howoRound, 0, 0, 1, 1, Kw, Kh, dilationW, dilationH, false, false, 0 });
        AscendC::LoadData(weightB2, weightB1, { 0, weRepeat, 1, 0, 0, false, 0 });

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

        AscendC::Mmad(dstCO1, featureMapA2, weightB2, { m, n, k, true, 0, false, false, false });

        outQueueCO1.EnQue<dstCO1_T>(dstCO1);
        inQueueFmA2.FreeTensor(featureMapA2);
        inQueueWeB2.FreeTensor(weightB2);
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

extern "C" __global__ __aicore__ void load3d_simple_kernel(__gm__ uint8_t *fmGm, __gm__ uint8_t *weGm,
    __gm__ uint8_t *biasGm, __gm__ uint8_t *dstGm)
{
    KernelLoad3d<dst_type, fmap_type, weight_type, dstCO1_type> op;
    op.Init(fmGm, weGm, biasGm, dstGm);
    op.Process();
}
```
