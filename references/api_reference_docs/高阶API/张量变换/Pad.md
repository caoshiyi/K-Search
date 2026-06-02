# Pad

**页面ID:** atlasascendc_api_07_0849  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0849.html

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

对height * width的二维Tensor在width方向上pad到32B对齐，如果Tensor的width已32B对齐，且全部为有效数据，则不支持调用本接口对齐。本接口具体功能场景如下：

- 场景1

Tensor的width非32B对齐，以half为例，如16*15，进行Pad，右边补1列，变成16*16。

- 场景2

Tensor的width已32B对齐，但是有部分冗余数据，以half为例，如16*16（最后两列为冗余数据），进行Pad，仍为16*16，但是最后两列冗余数据可以被填充成设定的值。

#### 函数原型

由于该接口的内部实现中涉及复杂的计算，需要额外的临时空间来存储计算过程中的中间变量。临时空间大小BufferSize的获取方法：通过Pad Tiling中提供的**GetPadMaxMinTmpSize**接口获取所需最大和最小临时空间大小，最小空间可以保证功能正确，最大空间用于提升性能。

临时空间支持**接口框架申请**和开发者**通过sharedTmpBuffer入参传入**两种方式，因此Pad接口的函数原型有两种：

- 通过sharedTmpBuffer入参传入临时空间

```
template <typename T>
__aicore__ inline void Pad(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor, PadParams& padParams, const LocalTensor<uint8_t>& sharedTmpBuffer, PadTiling& tiling)
```

该方式下开发者需自行申请并管理临时内存空间，并在接口调用完成后，复用该部分内存，内存不会反复申请释放，灵活性较高，内存利用率也较高。

- 接口框架申请临时空间

```
template <typename T>
__aicore__ inline void Pad(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor, PadParams& padParams, PadTiling& tiling)
```

该方式下开发者无需申请，但是需要预留临时空间的大小。

#### 参数说明

**表1 **模板参数说明

| 参数名 | 描述 |
| --- | --- |
| T | 操作数的数据类型。 Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持的数据类型为：int16_t、uint16_t、half、int32_t、uint32_t、float。 Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持的数据类型为：int16_t、uint16_t、half、int32_t、uint32_t、float。 Atlas 推理系列产品AI Core，支持的数据类型为：int16_t、uint16_t、half、int32_t、uint32_t、float。 |

**表2 **接口参数说明

| 参数名称 | 输入/输出 | 含义 |
| --- | --- | --- |
| dstTensor | 输出 | 目的操作数，shape为二维，LocalTensor数据结构的定义请参考LocalTensor。 类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| srcTensor | 输入 | 源操作数，shape为二维，LocalTensor数据结构的定义请参考LocalTensor。 类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| Pad参数，PadParams类型，PadParams结构体的具体参数如下： - leftPad，左边pad的数据量。单位：列。- rightPad，右边pad的数据量。单位：列。- padValue，补充的值，支持int32_t。 PadParams结构体的定义如下： ``` struct PadParams {     uint16_t leftPad = 0;     uint16_t rightPad = 0;     int32_t padValue = 0; }; ``` |  |  |
| sharedTmpBuffer | 输入 | 共享缓冲区，用于存放API内部计算产生的临时数据。该方式开发者可以自行管理sharedTmpBuffer内存空间，并在接口调用完成后，复用该部分内存，内存不会反复申请释放，灵活性较高，内存利用率也较高。共享缓冲区大小的获取方式请参考Pad Tiling。 类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| tiling | 输入 | 计算所需tiling信息，Tiling信息的获取请参考Pad Tiling。 |

#### 约束说明

- 对于场景1，支持同时在左边、右边进行pad。
- 对于场景2，只支持在右边进行pad。
- pad之后的总width不超过原width向最近32B对齐后的宽度。

#### 调用示例

本样例为场景1样例：Tensor的width非32B对齐，以half为例，如16*15，进行Pad，右边补1列，变成16*16。输入数据类型均为half。

```
#include "kernel_operator.h"

template <typename T>
class KernelPad {
public:
    __aicore__ inline KernelPad()
    {}
    __aicore__ inline void Init(GM_ADDR dstGm, GM_ADDR srcGm, uint32_t heightIn, uint32_t widthIn, uint32_t oriWidthIn,
        AscendC::PadParams &padParamsIn, const PadTiling &tilingData)
    {
        height = heightIn;
        width = widthIn;
        oriWidth = oriWidthIn;
        padParams = padParamsIn;
        srcGlobal.SetGlobalBuffer((__gm__ T *)srcGm);
        dstGlobal.SetGlobalBuffer((__gm__ T *)dstGm);
        pipe.InitBuffer(inQueueSrcVecIn, 1, height * width * sizeof(T));
        alignedWidth = ((width * sizeof(T) - 1) / 32 + 1) * 32 / sizeof(T);
        pipe.InitBuffer(inQueueSrcVecOut, 1, height * alignedWidth * sizeof(T));
        tiling = tilingData;
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
        AscendC::LocalTensor<T> srcLocal = inQueueSrcVecIn.AllocTensor<T>();
        AscendC::DataCopy(srcLocal, srcGlobal, height * width);
        inQueueSrcVecIn.EnQue(srcLocal);
    }
    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<T> dstLocal = inQueueSrcVecIn.DeQue<T>();
        AscendC::LocalTensor<T> srcOutLocal = inQueueSrcVecOut.AllocTensor<T>();
        AscendC::Pad(srcOutLocal, dstLocal, padParams, tiling);
        inQueueSrcVecOut.EnQue(srcOutLocal);
        inQueueSrcVecIn.FreeTensor(dstLocal);
    }
    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<T> srcOutLocalDe = inQueueSrcVecOut.DeQue<T>();
        AscendC::DataCopy(dstGlobal, srcOutLocalDe, height * alignedWidth);
        inQueueSrcVecOut.FreeTensor(srcOutLocalDe);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueSrcVecIn;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> inQueueSrcVecOut;
    AscendC::GlobalTensor<T> srcGlobal;
    AscendC::GlobalTensor<T> dstGlobal;
    uint32_t height;
    uint32_t width;
    uint32_t oriWidth;
    uint32_t alignedWidth;
    AscendC::PadParams padParams;
    PadTiling tiling;
};

extern "C" __global__ __aicore__ void kernel_pad_half_16_15_15(GM_ADDR src_gm, GM_ADDR dst_gm, __gm__ uint8_t *tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    KernelPad<half> op;
    AscendC::PadParams padParams{0, 1, 321};
    op.Init(dst_gm, src_gm, 16, 15, 15, padParams, tilingData.padTilingData);
    op.Process();
}
```

```
输入数据：
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14
15 16 17 18 19 20 21 22 23 24 25 26 27 28 29
30 31 32 33 34 35 36 37 38 39 40 41 42 43 44
45 46 47 48 49 50 51 52 53 54 55 56 57 58 59
60 61 62 63 64 65 66 67 68 69 70 71 72 73 74
75 76 77 78 79 80 81 82 83 84 85 86 87 88 89
90 91 92 93 94 95 96 97 98 99 100 101 102 103 104
105 106 107 108 109 110 111 112 113 114 115 116 117 118 119
120 121 122 123 124 125 126 127 128 129 130 131 132 133 134
135 136 137 138 139 140 141 142 143 144 145 146 147 148 149
150 151 152 153 154 155 156 157 158 159 160 161 162 163 164
165 166 167 168 169 170 171 172 173 174 175 176 177 178 179
180 181 182 183 184 185 186 187 188 189 190 191 192 193 194
195 196 197 198 199 200 201 202 203 204 205 206 207 208 209
210 211 212 213 214 215 216 217 218 219 220 221 222 223 224
225 226 227 228 229 230 231 232 233 234 235 236 237 238 239

输出数据
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 321
15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 321
30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 321
45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 321
60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 321
75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 321
90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 321
105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 321
120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 321
135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 321
150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 321
165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 321
180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 321
195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 321
210 211 212 213 214 215 216 217 218 219 220 221 222 223 224 321
225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 321
```
