# Sin

**页面ID:** atlasascendc_api_07_0500  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0500.html

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

按元素做正弦函数计算，计算公式如下：

<!-- img2text -->
```
Sin(x) = x - x³ / 3! + x⁵ / 5! - x⁷ / 7! + …
```

Sin(x)的泰勒展开式为：

![](images/atlasascendc_api_07_0500_img_002.png)

#### 函数原型

- 通过sharedTmpBuffer入参传入临时空间

  - 源操作数Tensor全部/部分参与计算

```
template<typename T, bool isReuseSource = false>
__aicore__ inline void Sin(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount)
```

  - 源操作数Tensor全部参与计算

```
template<typename T, bool isReuseSource = false>
__aicore__ inline void Sin(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor, const LocalTensor<uint8_t>& sharedTmpBuffer)
```

- 接口框架申请临时空间

  - 源操作数Tensor全部/部分参与计算

```
template<typename T, bool isReuseSource = false>
__aicore__ inline void Sin(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor, const uint32_t calCount)
```

  - 源操作数Tensor全部参与计算

```
template<typename T, bool isReuseSource = false>
__aicore__ inline void Sin(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor)
```

由于该接口的内部实现中涉及复杂的数学计算，需要额外的临时空间来存储计算过程中的中间变量。临时空间支持开发者**通过sharedTmpBuffer入参传入**和**接口框架申请**两种方式。

- 接口框架申请临时空间，开发者无需申请，但是需要预留临时空间的大小。

- 通过sharedTmpBuffer入参传入，使用该tensor作为临时空间进行处理，接口框架不再申请。该方式开发者可以自行管理sharedTmpBuffer内存空间，并在接口调用完成后，复用该部分内存，内存不会反复申请释放，灵活性较高，内存利用率也较高。

接口框架申请的方式，开发者需要预留临时空间；通过sharedTmpBuffer传入的情况，开发者需要为tensor申请空间。临时空间大小BufferSize的获取方式，通过GetSinMaxMinTmpSize接口获取需要预留空间的范围大小。

#### 参数说明

**表1 **模板参数说明

| 参数名 | 描述 |
| --- | --- |
| T | 操作数的数据类型。 Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持的数据类型为：half、float。 Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持的数据类型为：half、float。 Atlas 推理系列产品AI Core，支持的数据类型为：half、float。 |
| isReuseSource | 是否允许修改源操作数，默认值为false。该参数仅在输入的数据类型为float时生效。 - true：开发者允许源操作数被改写，可以使能该参数，使能后本接口内部计算时**复用**srcTensor的内存空间，节省部分内存空间；- false：本接口内部计算时**不复用**srcTensor的内存空间。 |

**表2 **参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| dstTensor | 输出 | 目的操作数。 类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| srcTensor | 输入 | 源操作数。 类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 源操作数的数据类型需要与目的操作数保持一致。 |
| sharedTmpBuffer | 输入 | 临时缓存。 类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 用于Sin内部复杂计算时存储中间变量，由开发者提供。 临时空间大小BufferSize的获取方式请参考GetSinMaxMinTmpSize。 |
| calCount | 输入 | 参与计算的元素个数。 |

#### 约束说明

- 对于以下产品，输入源数据必须保持值域在[-65504.0, 65504.0]范围内。

  - Atlas A3 训练系列产品/Atlas A3 推理系列产品
  - Atlas A2 训练系列产品/Atlas A2 推理系列产品
  - Atlas 推理系列产品AI Core

- **不支持源操作数与目的操作数地址重叠。**
- 不支持sharedTmpBuffer与源操作数和目的操作数地址重叠。

#### 调用示例

```
AscendC::TPipe pipe;
AscendC::TQue<AscendC::TPosition::VECCALC, 1> tmpQue;
pipe.InitBuffer(tmpQue, 1, bufferSize); // bufferSize 通过Host侧tiling参数获取
AscendC::LocalTensor<uint8_t> sharedTmpBuffer = tmpQue.AllocTensor<uint8_t>();
// 输入tensor长度为1024, 算子输入的数据类型为half, 实际计算个数为512
AscendC::Sin(dstLocal, srcLocal, sharedTmpBuffer, 512);
```

结果示例如下：

```
输入数据(srcLocal): 
[-0.44476402 -0.43156096 -0.38648438  0.30285975 -0.73223037 -0.57837343 ... -0.25575528  0.5976324 ]
输出数据(dstLocal): 
[-0.43024486 -0.41828915 -0.37693444  0.29825103 -0.66853    -0.5466626 ... -0.25297618  0.56268686]
```
