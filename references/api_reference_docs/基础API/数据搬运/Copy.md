# Copy

**页面ID:** atlasascendc_api_07_0106  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0106.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品            /             Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品            /             Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品            AI Core | x |
| Atlas 推理系列产品            Vector Core | x |
| Atlas 训练系列产品 | x |

#### 功能说明

VECIN，VECCALC，VECOUT之间的搬运指令，支持mask操作和DataBlock间隔操作。

#### 函数原型

- tensor高维切分计算

  - mask逐bit模式

```
template <typename T, bool isSetMask = true>
__aicore__ inline void Copy(const LocalTensor<T>& dst, const LocalTensor<T>& src, const uint64_t mask[], const uint8_t repeatTime, const CopyRepeatParams& repeatParams)
```

  - mask连续模式

```
template <typename T, bool isSetMask = true>
__aicore__ inline void Copy(const LocalTensor<T>& dst, const LocalTensor<T>& src, const uint64_t mask, const uint8_t repeatTime, const CopyRepeatParams& repeatParams)
```

#### 参数说明

**表1 **模板参数说明

| 参数名 | 描述 |
| --- | --- |
| T | 操作数数据类型。                       Atlas A3 训练系列产品            /             Atlas A3 推理系列产品            ，支持的数据类型为：uint16_t/int16_t/half/bfloat16_t/uint32_t/int32_t/float                       Atlas A2 训练系列产品            /             Atlas A2 推理系列产品            ，支持的数据类型为：uint16_t/int16_t/half/bfloat16_t/uint32_t/int32_t/float                       Atlas 200I/500 A2 推理产品            ，支持的数据类型为：uint16_t/int16_t/half/bfloat16_t/uint32_t/int32_t/float |
| isSetMask | 是否在接口内部设置mask。                     - true，表示在接口内部设置mask。           - false，表示在接口外部设置mask，开发者需要使用SetVectorMask接口设置mask值。这种模式下，本接口入参中的mask值必须设置为占位符MASK_PLACEHOLDER。 |

**表2 **参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| dst | 输出 | 目的操作数。          类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。起始地址需要保证32字节对齐。 |
| src | 输入 | 源操作数。          类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。起始地址需要保证32字节对齐。          源操作数的数据类型需要与目的操作数保持一致。 |
| mask/mask[] | 输入 | mask用于控制每次迭代内参与计算的元素。                     - 逐bit模式：可以按位控制哪些元素参与计算，bit位的值为1表示参与计算，0表示不参与。            mask为数组形式，数组长度和数组元素的取值范围和操作数的数据类型有关。当操作数为16位时，数组长度为2，mask[0]、mask[1]∈[0, 264-1]并且不同时为0；当操作数为32位时，数组长度为1，mask[0]∈(0, 264-1]；当操作数为64位时，数组长度为1，mask[0]∈(0, 232-1]。            例如，mask=[8, 0]，8=0b1000，表示仅第4个元素参与计算。                               - 连续模式：表示前面连续的多少个元素参与计算。取值范围和操作数的数据类型有关，数据类型不同，每次迭代内能够处理的元素个数最大值不同。当操作数为16位时，mask∈[1, 128]；当操作数为32位时，mask∈[1, 64]；当操作数为64位时，mask∈[1, 32]。 |
| repeatTime | 输入 | 重复迭代次数。矢量计算单元，每次读取连续的256Bytes数据进行计算，为完成对输入数据的处理，必须通过多次迭代（repeat）才能完成所有数据的读取与计算。repeatTime表示迭代的次数。          关于该参数的具体描述请参考高维切分API。 |
| repeatParams | 输入 | 控制操作数地址步长的数据结构。CopyRepeatParams类型。          具体定义请参考${INSTALL_DIR}/include/ascendc/basic_api/interface/kernel_struct_data_copy.h，${INSTALL_DIR}请替换为CANN软件安装后文件存储路径。          参数说明请参考表3。 |

**表3 **CopyRepeatParams结构体参数说明

| 参数名称 | 含义 |
| --- | --- |
| dstStride、srcStride | 用于设置同一迭代内datablock的地址步长，取值范围为[0,65535]。          同一迭代内datablock的地址步长参数说明请参考dataBlockStride。 |
| dstRepeatSize、srcRepeatSize | 用于设置相邻迭代间的地址步长，取值范围为[0,4095]。          相邻迭代间的地址步长参数说明请参考repeatStride。 |

#### 约束说明

- 源操作数和目的操作数的起始地址需要保证32字节对齐。
- Copy和矢量计算API一样，支持和掩码操作API配合使用。但Counter模式配合高维切分计算API时，和通用的Counter模式有一定差异。具体差异如下：

  - 通用的Counter模式：Mask代表**整个矢量计算参与计算的元素个数，迭代次数不生效**。
  - Counter模式配合Copy高维切分计算API，Mask代表**每次Repeat中处理的元素个数，迭代次数生效。**示意图如下：

<!-- img2text -->
```
srcLocal
┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┐
│ 1  │    │ 2  │    │ 3  │ ...│ N  │ ...│ 1  │    │ 2  │    │ 3  │ ...│ N  │    │
└────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┘
<───────────────────────────────────────>
              srcRepeatSize

<────>
srcStride

<───>
 bk1

        <───>
         bk2

                <───>
                 bk3

                        <───>
                         bkN

<──────────────────────────────>
          repeat 1

                                        <───>
                                         bk1

                                                <───>
                                                 bk2

                                                        <───>
                                                         bk3

                                                                        <───>
                                                                         bkN

                                        <──────────────────────────────>
                                                  repeat 2


N=ceil(Mask*sizeof(T)/datablockSize)


dstLocal
┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┐
│ 1  │    │ 2  │    │ 3  │ ...│ N  │ ...│ 1  │ 2  │ 3  │ ...│ N  │    │    │
└────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┘
<──────────────────────────────────>
            dstRepeatSize

<──>
dstStride

<───>
 bk1

      <───>
       bk2

            <───>
             bk3

                    <───>
                     bkN

<───────────────────────>
        repeat 1

                                <───>
                                 bk1

                                      <───>
                                       bk2

                                            <───>
                                             bk3

                                                    <───>
                                                     bkN

                                <───────────────────────>
                                          repeat 2
```

- srcRepeatSize: srcLocal 覆盖第1-8块(从第1个 repeat 的起始到第2个 repeat 的起始前)
- srcStride: srcLocal 覆盖第1-2块(相邻源块之间的间隔)
- bk1: srcLocal 覆盖第1-1块(第1个数据块)
- bk2: srcLocal 覆盖第3-3块(第2个数据块)
- bk3: srcLocal 覆盖第5-5块(第3个数据块)
- bkN: srcLocal 覆盖第7-7块(第N个数据块)
- repeat 1: srcLocal 覆盖第1-7块(第1次 Repeat 的处理范围)
- bk1: srcLocal 覆盖第9-9块(第2个 repeat 的第1个数据块)
- bk2: srcLocal 覆盖第11-11块(第2个 repeat 的第2个数据块)
- bk3: srcLocal 覆盖第13-13块(第2个 repeat 的第3个数据块)
- bkN: srcLocal 覆盖第15-15块(第2个 repeat 的第N个数据块)
- repeat 2: srcLocal 覆盖第9-15块(第2次 Repeat 的处理范围)

- dstRepeatSize: dstLocal 覆盖第1-8块(从第1个 repeat 的起始到第2个 repeat 的起始前)
- dstStride: dstLocal 覆盖第1-1块(相邻目标块之间的间隔)
- bk1: dstLocal 覆盖第1-1块(第1个数据块)
- bk2: dstLocal 覆盖第3-3块(第2个数据块)
- bk3: dstLocal 覆盖第5-5块(第3个数据块)
- bkN: dstLocal 覆盖第7-7块(第N个数据块)
- repeat 1: dstLocal 覆盖第1-7块(第1次 Repeat 的处理范围)
- bk1: dstLocal 覆盖第9-9块(第2个 repeat 的第1个数据块)
- bk2: dstLocal 覆盖第10-10块(第2个 repeat 的第2个数据块)
- bk3: dstLocal 覆盖第11-11块(第2个 repeat 的第3个数据块)
- bkN: dstLocal 覆盖第13-13块(第2个 repeat 的第N个数据块)
- repeat 2: dstLocal 覆盖第9-13块(第2次 Repeat 的处理范围)

#### 调用示例

本示例仅展示Compute流程中的部分代码。如需运行，请参考样例模板实现完整的代码。

本示例中操作数数据类型为int16_t。

- mask连续模式

```
uint64_t mask = 128;
// repeatTime = 4, 128 elements one repeat, 512 elements total
// dstStride, srcStride = 1, no gap between blocks in one repeat
// dstRepStride, srcRepStride = 8, no gap between repeats
AscendC::Copy(dstLocal, srcLocal, mask, 4, { 1, 1, 8, 8 });
```

结果示例如下：

```
输入数据srcLocal： [9 -2 8 ... 9]
输出数据dstLocal： 
[9 -2 8 ... 9]
```

- mask逐bit模式

```
uint64_t mask[2] = { UINT64_MAX, UINT64_MAX };
// repeatTime = 4, 128 elements one repeat, 512 elements total
// dstStride, srcStride = 1, no gap between blocks in one repeat
// dstRepStride, srcRepStride = 8, no gap between repeats
AscendC::Copy(dstLocal, srcLocal, mask, 4, { 1, 1, 8, 8 });
```

结果示例如下：

```
输入数据srcLocal：[9 -2 8 ... 9]
输出数据dstLocal： 
[9 -2 8 ... 9]
```

#### 样例模板

```
#include "kernel_operator.h"
class KernelCopy {
public:
    __aicore__ inline KernelCopy() {}
    __aicore__ inline void Init(__gm__ uint8_t* srcGm, __gm__ uint8_t* dstGm)
    {
        srcGlobal.SetGlobalBuffer((__gm__ int32_t*)srcGm);
        dstGlobal.SetGlobalBuffer((__gm__ int32_t*)dstGm);
        pipe.InitBuffer(inQueueSrc, 1, 512 * sizeof(int32_t));
        pipe.InitBuffer(outQueueDst, 1, 512 * sizeof(int32_t));
    }
    __aicore__ inline void Process()
    {
        CopyIn();
        Compute();
        CopyOut();
    }
private:
    __aicore__ inline void CopyIn()
    {
        AscendC::LocalTensor<int32_t> srcLocal = inQueueSrc.AllocTensor<int32_t>();
        AscendC::DataCopy(srcLocal, srcGlobal, 512);
        inQueueSrc.EnQue(srcLocal);
    }
    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<int32_t> srcLocal = inQueueSrc.DeQue<int32_t>();
        AscendC::LocalTensor<int32_t> dstLocal = outQueueDst.AllocTensor<int32_t>();
        uint64_t mask = 64;
        AscendC::Copy(dstLocal, srcLocal, mask, 4, { 1, 1, 8, 8 });
        outQueueDst.EnQue<int32_t>(dstLocal);
        inQueueSrc.FreeTensor(srcLocal);
    }
    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<int32_t> dstLocal = outQueueDst.DeQue<int32_t>();
        AscendC::DataCopy(dstGlobal, dstLocal, 512);
        outQueueDst.FreeTensor(dstLocal);
    }
private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueSrc;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueDst;
    AscendC::GlobalTensor<int32_t> srcGlobal, dstGlobal;
};
extern "C" __global__ __aicore__ void copy_simple_kernel(__gm__ uint8_t* srcGm, __gm__ uint8_t* dstGm)
{
    KernelCopy op;
    op.Init(srcGm, dstGm);
    op.Process();
}
```
