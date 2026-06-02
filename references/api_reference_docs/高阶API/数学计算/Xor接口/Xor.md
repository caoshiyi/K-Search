# Xor

**页面ID:** atlasascendc_api_07_0601  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0601.html

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

按元素执行Xor运算，Xor（异或）的概念和运算规则如下：

- 概念：参加运算的两个数据，按二进制位进行“异或”运算。
- 运算规则：0^0=0；0^1=1；1^0=1；1^1=0；即：参加运算的两个对象，如果两个相应位为“异”（值不同），则该位结果为1，否则为 0【同0异1】。

计算公式如下：

<!-- img2text -->
```
例如：3^5=6，即0000 0011^0000 0101 = 0000 0110
```

<!-- img2text -->
```
A xor B = C

┌────────┐   ┌────────┐        ┌────────┐
│   A    │   │   B    │        │   C    │
└────────┘   └────────┘        └────────┘
     │            │                ▲
     └────────────┴──── xor ───────┘
```

```
例如：3^5=6，即0000 0011^0000 0101 = 0000 0110
```

#### 函数原型

- 通过sharedTmpBuffer入参传入临时空间

  - 源操作数Tensor全部/部分参与计算

```
template <typename T, bool isReuseSource = false>
__aicore__ inline void Xor(const LocalTensor<T>& dstTensor, const LocalTensor<T>& src0Tensor, const LocalTensor<T>& src1Tensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount)
```

  - 源操作数Tensor全部参与计算

```
template <typename T, bool isReuseSource = false>
__aicore__ inline void Xor(const LocalTensor<T>& dstTensor, const LocalTensor<T>& src0Tensor, const LocalTensor<T>& src1Tensor, const LocalTensor<uint8_t>& sharedTmpBuffer)
```

- 接口框架申请临时空间

  - 源操作数Tensor全部/部分参与计算

```
template <typename T, bool isReuseSource = false>
__aicore__ inline void Xor(const LocalTensor<T>& dstTensor, const LocalTensor<T>& src0Tensor, const LocalTensor<T>& src1Tensor, const uint32_t calCount)
```

  - 源操作数Tensor全部参与计算

```
template <typename T, bool isReuseSource = false>
__aicore__ inline void Xor(const LocalTensor<T>& dstTensor, const LocalTensor<T>& src0Tensor, const LocalTensor<T>& src1Tensor)
```

由于该接口的内部实现中涉及复杂的数学计算，需要额外的临时空间来存储计算过程中的中间变量。临时空间支持开发者**通过sharedTmpBuffer入参传入**和**接口框架申请**两种方式。

- 通过sharedTmpBuffer入参传入，使用该tensor作为临时空间进行处理，接口框架不再申请。该方式开发者可以自行管理sharedTmpBuffer内存空间，并在接口调用完成后，复用该部分内存，内存不会反复申请释放，灵活性较高，内存利用率也较高。
- 接口框架申请临时空间，开发者无需申请，但是需要预留临时空间的大小。

通过sharedTmpBuffer传入的情况，开发者需要为tensor申请空间；接口框架申请的方式，开发者需要预留临时空间。临时空间大小BufferSize的获取方式如下：通过GetXorMaxMinTmpSize中提供的接口获取需要预留空间范围的大小。

#### 参数说明

**表1 **模板参数说明

| 参数名 | 描述 |
| --- | --- |
| T | 操作数的数据类型。 Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持的数据类型为：int16_t、uint16_t。 Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持的数据类型为：int16_t、uint16_t。 Atlas 200I/500 A2 推理产品，支持的数据类型为：int16_t、uint16_t。 Atlas 推理系列产品AI Core，支持的数据类型为：int16_t、uint16_t。 |
| isReuseSource | 是否允许修改源操作数。该参数预留，传入默认值false即可。 |

**表2 **接口参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| dstTensor | 输出 | 目的操作数。 类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| src0Tensor | 输入 | 源操作数0。 类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 源操作数的数据类型需要与目的操作数保持一致。 |
| src1Tensor | 输入 | 源操作数1。 类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 源操作数的数据类型需要与目的操作数保持一致。 |
| sharedTmpBuffer | 输入 | 临时缓存。 类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 用于Xor内部复杂计算时存储中间变量，由开发者提供。 临时空间大小BufferSize的获取方式请参考GetXorMaxMinTmpSize。 |
| calCount | 输入 | 参与计算的元素个数。 |

#### 约束说明

- **不支持源操作数与目的操作数地址重叠。**
- 当前仅支持ND格式的输入，不支持其他格式。
- calCount需要保证小于或等于src0Tensor和src1Tensor和dstTensor存储的元素范围。
- 对于不带calCount参数的接口，需要保证src0Tensor和src1Tensor的shape大小相等。
- 不支持sharedTmpBuffer与源操作数和目的操作数地址重叠。

#### 调用示例

**调用样例kernel侧****xor_custom.cpp**

```
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 1;
class KernelXor {
public:
    __aicore__ inline KernelXor() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength, uint32_t totalLength2, uint32_t tilenum, uint32_t tmpSize, uint32_t mcount)
    {
        this->totalLength = totalLength;
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->blockLength2 = totalLength2 / AscendC::GetBlockNum();
        this->tilenum = tilenum;
        this->tmpSize = tmpSize;
        this->mcount = mcount;
        this->tileLength = this->blockLength / tilenum / BUFFER_NUM;
        this->tileLength2 = this->blockLength2 / tilenum / BUFFER_NUM;

        xGm.SetGlobalBuffer((__gm__ int16_t *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ int16_t *)y + this->blockLength2 * AscendC::GetBlockIdx(), this->blockLength2);
        zGm.SetGlobalBuffer((__gm__ int16_t *)z + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);

        if (this->tmpSize != 0) {
            pipe.InitBuffer(tmpQueue, BUFFER_NUM, this->tmpSize);
        }
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(int16_t));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength2 * sizeof(int16_t));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(int16_t));
    }
    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tilenum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<int16_t> xLocal = inQueueX.AllocTensor<int16_t>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
        AscendC::LocalTensor<int16_t> yLocal = inQueueY.AllocTensor<int16_t>();
        AscendC::DataCopy(yLocal, yGm[progress * this->tileLength2], this->tileLength2);
        inQueueY.EnQue(yLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<int16_t> xLocal = inQueueX.DeQue<int16_t>();
        AscendC::LocalTensor<int16_t> yLocal = inQueueY.DeQue<int16_t>();
        AscendC::LocalTensor<int16_t> zLocal = outQueueZ.AllocTensor<int16_t>();
        if (this->tmpSize != 0) {
            AscendC::LocalTensor<uint8_t> tmpLocal = tmpQueue.AllocTensor<uint8_t>();
            if (this->mcount != this->totalLength) {
                AscendC::Xor(zLocal, xLocal, yLocal, tmpLocal, this->mcount);
            } else {
                AscendC::Xor(zLocal, xLocal, yLocal, tmpLocal);
            }
            tmpQueue.FreeTensor(tmpLocal);
        } else {
            if (this->mcount != this->totalLength) {
                AscendC::Xor(zLocal, xLocal, yLocal, this->mcount);
            } else {
                AscendC::Xor(zLocal, xLocal, yLocal);
            }
        }
        outQueueZ.EnQue<int16_t>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<int16_t> zLocal = outQueueZ.DeQue<int16_t>();
        AscendC::DataCopy(zGm[progress * this->tileLength], zLocal, this->tileLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueY;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> tmpQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<int16_t> xGm;
    AscendC::GlobalTensor<int16_t> yGm;
    AscendC::GlobalTensor<int16_t> zGm;
    uint32_t blockLength;
    uint32_t blockLength2;
    uint32_t tilenum;
    uint32_t tileLength;
    uint32_t tileLength2;
    uint32_t tmpSize;
    uint32_t mcount;
    uint32_t totalLength;
};

extern "C" __global__ __aicore__ void xor_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    KernelXor op;
    op.Init(x, y, z, tilingData.totalLength, tilingData.totalLength2, tilingData.tilenum, tilingData.tmpSize, tilingData.mcount);
    if (TILING_KEY_IS(1)) {
        op.Process();
    }
}
```

**host侧xor_custom_tiling.h**

```
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
namespace optiling {
  BEGIN_TILING_DATA_DEF(XorCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalLength);
  TILING_DATA_FIELD_DEF(uint32_t, totalLength2);
  TILING_DATA_FIELD_DEF(uint32_t, tmpSize);
  TILING_DATA_FIELD_DEF(uint32_t, tilenum);
  TILING_DATA_FIELD_DEF(uint32_t, mcount);
  END_TILING_DATA_DEF;
  REGISTER_TILING_DATA_CLASS(XorCustom, XorCustomTilingData)
}
```

**host侧****xor_custom.cpp**

```
#include "xor_custom_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"

namespace optiling
{
    static ge::graphStatus TilingFunc(gert::TilingContext *context)
    {
        XorCustomTilingData tiling;
        const gert::RuntimeAttrs *xorAttrs = context->GetAttrs();
        const uint32_t tilenum = *(xorAttrs->GetAttrPointer<uint32_t>(0));
        const uint32_t blockdim = *(xorAttrs->GetAttrPointer<uint32_t>(1));
        const uint32_t sizeflag = *(xorAttrs->GetAttrPointer<uint32_t>(2));
        const uint32_t countflag = *(xorAttrs->GetAttrPointer<uint32_t>(3));
        uint32_t totalLength = context->GetInputTensor(0)->GetShapeSize();
        uint32_t totalLength2 = context->GetInputTensor(1)->GetShapeSize();
        context->SetBlockDim(blockdim);
        tiling.set_totalLength(totalLength);
        tiling.set_totalLength2(totalLength2);
        tiling.set_tilenum(tilenum);

        if (countflag == 0) {
            tiling.set_mcount(totalLength2);
        } else if (countflag == 1) {
            tiling.set_mcount(totalLength);
        }

        std::vector<int64_t> shapeVec = {totalLength};
        ge::Shape srcShape(shapeVec);
        uint32_t typeSize = sizeof(int16_t);
        uint32_t maxValue = 0;
        uint32_t minValue = 0;
        bool isReuseSource = false;
        AscendC::GetXorMaxMinTmpSize(srcShape, typeSize, isReuseSource, maxValue, minValue);
        // sizeflag 0：代表取最小的tempBuffer 1：取最大的tempBuffer
        if (sizeflag == 0) {
            tiling.set_tmpSize(minValue);
        } else if (sizeflag == 1) {
            tiling.set_tmpSize(maxValue);
        } else if (sizeflag == 2) {
            tiling.set_tmpSize(0);
        }
        tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
        context->SetTilingKey(1);
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}
namespace ge
{
    static ge::graphStatus InferShape(gert::InferShapeContext *context)
    {
        const gert::Shape *xShape = context->GetInputShape(0);
        gert::Shape *yShape = context->GetOutputShape(0);
        *yShape = *xShape;
        return GRAPH_SUCCESS;
    }
}
namespace ops
{
    class XorCustom : public OpDef
    {
    public:
        explicit XorCustom(const char *name) : OpDef(name)
        {
            this->Input("x")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT16})
                .Format({ge::FORMAT_ND});
            this->Input("y")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT16})
                .Format({ge::FORMAT_ND});
            this->Output("z")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT16})
                .Format({ge::FORMAT_ND});
            this->SetInferShape(ge::InferShape);
            this->Attr("tilenum")
                .AttrType(REQUIRED)
                .Int(0);
            this->Attr("blockdim")
                .AttrType(REQUIRED)
                .Int(0);
            this->Attr("sizeflag")
                .AttrType(REQUIRED)
                .Int(0);
            this->Attr("countflag")
                .AttrType(REQUIRED)
                .Int(0);
            this->AICore()
                .SetTiling(optiling::TilingFunc);
            this->AICore().AddConfig("ascendxxx"); // ascendxxx请修改为对应的昇腾AI处理器型号。
        }
    };
    OP_ADD(XorCustom);
} // namespace ops
```

结果示例如下：

```
输入输出的数据类型为int16_t，一维向量包含32个数。例如向量中第一个数据进行异或：(-5753) xor 18745 = -24386
输入数据(src0Local): [-5753 28501 20334 -5845  ... -20817 3403 21261 22241]
输入数据(src1Local): [18745 -24448 20873 10759 ... 21940 -26342 9251 31019]
输出数据(dstLocal): [-24386 -12331 7911 -15572 ... -1253 -27567 30510 12234]
```
