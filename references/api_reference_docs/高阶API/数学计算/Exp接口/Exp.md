# Exp

**页面ID:** atlasascendc_api_07_0589  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0589.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品            /             Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品            /             Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | x |
| Atlas 推理系列产品            AI Core | x |
| Atlas 推理系列产品            Vector Core | x |
| Atlas 训练系列产品 | x |

#### 功能说明

按元素取自然指数，用户可以选择是否使用泰勒展开公式进行计算，计算公式如下：

<!-- img2text -->
```
      x
y = e
```

- 设置泰勒展开项数为0，即不使用泰勒展开公式进行计算，公式如下：

![](images/atlasascendc_api_07_0589_img_002.png)

- 设置泰勒展开项数不为0，即使用泰勒展开公式进行计算，公式如下：

![](images/atlasascendc_api_07_0589_img_003.png)

xAi代表源操作数的整数部分，该值通过floor(x)获取。xBi代表源操作数的小数部分。

泰勒展开公式如下：

![](images/atlasascendc_api_07_0589_img_004.png)

#### 函数原型

- 通过sharedTmpBuffer入参传入临时空间

```
template <typename T, uint8_t taylorExpandLevel, bool isReuseSource = false>
__aicore__ inline void Exp(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount)
```

- 接口框架申请临时空间

```
template <typename T, uint8_t taylorExpandLevel, bool isReuseSource = false>
__aicore__ inline void Exp(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const uint32_t calCount)
```

由于该接口的内部实现中涉及复杂的数学计算，需要额外的临时空间来存储计算过程中的中间变量。临时空间支持开发者**通过sharedTmpBuffer入参传入**和**接口框架申请**两种方式。

- 通过sharedTmpBuffer入参传入，使用该tensor作为临时空间进行处理，接口框架不再申请。该方式开发者可以自行管理sharedTmpBuffer内存空间，并在接口调用完成后，复用该部分内存，内存不会反复申请释放，灵活性较高，内存利用率也较高。
- 接口框架申请临时空间，开发者无需申请，但是需要预留临时空间的大小。

通过sharedTmpBuffer传入的情况，开发者需要为tensor申请空间；接口框架申请的方式，开发者需要预留临时空间。临时空间大小BufferSize的获取方式如下：通过GetExpMaxMinTmpSize中提供的接口获取需要预留空间范围的大小。

#### 参数说明

**表1 **模板参数说明

| 参数名 | 描述 |
| --- | --- |
| T | 操作数的数据类型。                       Atlas A3 训练系列产品            /             Atlas A3 推理系列产品            ，支持的数据类型为：half、float。                       Atlas A2 训练系列产品            /             Atlas A2 推理系列产品            ，支持的数据类型为：half、float。 |
| taylorExpandLevel | 泰勒展开项数，项数为0表示不使用泰勒公式进行计算。项数太少时，精度会有一定误差。项数越多，精度相对而言更高，但是性能会更差。取值范围为[0, 255]，推荐取值为[10, 15] |
| isReuseSource | 是否允许修改源操作数，默认值为false。该参数仅在输入的数据类型为float时生效。                     - true：开发者允许源操作数被改写，可以使能该参数，使能后本接口内部计算时**复用**srcLocal的内存空间，节省部分内存空间；           - false：本接口内部计算时**不复用**srcLocal的内存空间。                    isReuseSource的使用样例请参考更多样例。 |

**表2 **接口参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| dstLocal | 输出 | 目的操作数。          类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| srcLocal | 输入 | 源操作数。          类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。          源操作数的数据类型需要与目的操作数保持一致。 |
| sharedTmpBuffer | 输入 | 临时缓存。          类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。          用于Exp内部复杂计算时存储中间变量，由开发者提供。          临时空间大小BufferSize的获取方式请参考GetExpMaxMinTmpSize。 |
| calCount | 输入 | 参与计算的元素个数。 |

#### 约束说明

- **不支持源操作数与目的操作数地址重叠。**
- 不支持sharedTmpBuffer与源操作数和目的操作数地址重叠。

#### 调用示例

完整的调用样例请参考更多样例。

```
AscendC::TPipe pipe;
AscendC::TQue<AscendC::TPosition::VECCALC, 1> tmpQue;
pipe.InitBuffer(tmpQue, 1, bufferSize);  // bufferSize 通过Host侧tiling参数获取
AscendC::LocalTensor<uint8_t> sharedTmpBuffer = tmpQue.AllocTensor<uint8_t>();
// 输入tensor长度为1024, 算子输入的数据类型为half, 实际计算个数为512
AscendC::Exp<half, 15, false>(dstLocal, srcLocal, sharedTmpBuffer, 512);
```
