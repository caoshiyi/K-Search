# Brcb

**页面ID:** atlasascendc_api_07_0089  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0089.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | x |
| Atlas 推理系列产品AI Core | √ |
| Atlas 推理系列产品Vector Core | x |
| Atlas 训练系列产品 | x |

#### 功能说明

给定一个输入张量，每一次取输入张量中的8个数填充到结果张量的8个datablock（32Bytes）中去，每个数对应一个datablock。

#### 函数原型

```
template <typename T>
__aicore__ inline void Brcb(const LocalTensor<T>& dst, const LocalTensor<T>& src0, const uint8_t repeatTime, const BrcbRepeatParams& repeatParams)
```

#### 参数说明

**表1 **模板参数说明

| 参数名 | 描述 |
| --- | --- |
| T | 操作数数据类型。 Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持的数据类型为：int16_t/uint16_t/int32_t/uint32_t/half/bfloat16_t/float Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持的数据类型为：int16_t/uint16_t/int32_t/uint32_t/half/bfloat16_t/float Atlas 推理系列产品AI Core，支持的数据类型为：int16_t/uint16_t/int32_t/uint32_t/half/float |

**表2 **参数说明

| 参数名称 | 输入/输出 | 含义 |
| --- | --- | --- |
| dst | 输出 | 目的操作数。 类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 LocalTensor的起始地址需要32字节对齐。 |
| src0 | 输入 | 源操作数。 类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 LocalTensor的起始地址需要32字节对齐。 数据类型和dst保持一致。 每一次迭代读取src0中的8个元素，所以src0的元素个数不小于8 * repeatTime。 |
| repeatTime | 输入 | 指令迭代次数，每次迭代完成8个datablock的数据收集，数据范围：repeatTime∈[0,255]。 |
| repeatParams | 输入 | 用于控制指令迭代的相关参数。 类型为BrcbRepeatParams，具体定义可参考${INSTALL_DIR}/include/ascendc/basic_api/interface/kernel_struct_brcb.h。${INSTALL_DIR}请替换为CANN软件安装后文件存储路径。 其中dstBlkStride、dstRepStride支持用户配置，参数说明参考表3。 |

**表3 **BrcbRepeatParams结构体参数说明

| 参数名称 | 含义 |
| --- | --- |
| dstBlkStride | 单次迭代内，矢量目的操作数不同datablock间地址步长。 **注意事项：** 当dstBlkStride值为0时，默认按照1来处理。 |
| dstRepStride | 相邻迭代间，矢量目的操作数相同datablock地址步长。 |
| blockNumber | 预留参数。为后续的功能做保留，开发者暂时无需关注，使用默认值即可。 |
| src0BlkStride |  |
| src1BlkStride |  |
| src0RepStride |  |
| src1RepStride |  |
| repeatStrideMode |  |
| strideSizeMode |  |

#### 约束说明

- 不支持src0与dst为同一块内存地址。
- 针对Atlas 推理系列产品AI Core，使用时需要预留8K的Unified Buffer空间，作为接口的临时数据存放区。

#### 调用示例

uint16_t数据类型brcb示例

```
#include "kernel_operator.h"
class VbrcbCase {
public:
    __aicore__ inline VbrcbCase()
    {}
    __aicore__ inline void Init(__gm__ uint8_t *x, __gm__ uint8_t *y)
    {
        x_gm.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t *>(x));
        y_gm.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t *>(y));
        tpipe.InitBuffer(vecIn, 1, 16 * sizeof(uint16_t));
        tpipe.InitBuffer(vecOut, 1, 256 * sizeof(uint16_t));
    }
    __aicore__ inline void Process()
    {
        CopyIn();
        Compute();
        CopyOut();
    }
    __aicore__ inline void CopyIn()
    {
        auto x_buf = vecIn.AllocTensor<uint16_t>();
        AscendC::DataCopy(x_buf, x_gm, 16);
        vecIn.EnQue(x_buf);
    }
    __aicore__ inline void Compute()
    {
        auto x_buf = vecIn.DeQue<uint16_t>();
        auto y_buf = vecOut.AllocTensor<uint16_t>();
        AscendC::Brcb(y_buf, x_buf, 2, {1,8});
        vecOut.EnQue(y_buf);
        vecIn.FreeTensor(x_buf);
    }
    __aicore__ inline void CopyOut()
    {
        auto y_buf = vecOut.DeQue<uint16_t>();
        AscendC::DataCopy(y_gm, y_buf, 256);
        vecOut.FreeTensor(y_buf);
    }
private:
    AscendC::GlobalTensor<uint16_t> x_gm;
    AscendC::GlobalTensor<uint16_t> y_gm;
    AscendC::TPipe tpipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> vecIn;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> vecOut;
};
extern "C" __global__ __aicore__ void vbrcb_uint16_t_16(__gm__ uint8_t *x, __gm__ uint8_t *y)
{
    VbrcbCase op;
    op.Init(x, y);
    op.Process();
}
```

结果示例：

```
输入数据x_gm：[1 2 3 ... 16]
输出数据y_gm：[1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 ... 15 15 15 15 15 15 15 15 15 15 15 15 15 15 15 15 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16]
```
