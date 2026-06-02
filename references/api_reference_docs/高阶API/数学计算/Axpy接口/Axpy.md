# Axpy

**页面ID:** atlasascendc_api_07_0585  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0585.html

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

源操作数(srcTensor)中每个元素与标量求积后和目的操作数(dstTensor)中的对应元素相加，计算公式如下：

![](images/atlasascendc_api_07_0585_img_001.png)

![](images/atlasascendc_api_07_0585_img_002.png)

该接口功能同基础API Axpy，区别在于此接口指令是通过Muls和Add组合计算，从而提供更优的精度。

#### 函数原型

```
template <typename T, typename U, bool isReuseSource = false>
__aicore__ inline void Axpy(const LocalTensor<T>& dstTensor, const LocalTensor<U>& srcTensor, const U scalarValue, const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount)
```

#### 参数说明

**表1 **模板参数说明

| 参数名 | 描述 |
| --- | --- |
| T | 目的操作数的数据类型。 Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持的数据类型为：half、float。 Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持的数据类型为：half、float。 Atlas 推理系列产品AI Core，支持的数据类型为：half、float。 |
| U | 源操作数的数据类型。 Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持的数据类型为：half、float。 Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持的数据类型为：half、float。 Atlas 推理系列产品AI Core，支持的数据类型为：half、float。 |
| isReuseSource | 是否允许修改源操作数。该参数预留，传入默认值false即可。 |

**表2 **接口参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| dstTensor | 输出 | 目的操作数。 类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| srcTensor | 输入 | 源操作数。 类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| scalarValue | 输入 | scalar标量。支持的数据类型为：half、float。scalar操作数的类型需要和srcTensor保持一致。 |
| sharedTmpBuffer | 输入 | 临时缓存。 类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 由于该接口的内部实现中涉及复杂的数学计算，需要额外的临时空间来存储计算过程中的中间变量。临时空间需要开发者通过sharedTmpBuffer入参传入。临时空间大小BufferSize的获取方式请参考GetAxpyMaxMinTmpSize。 |
| calCount | 输入 | 参与计算的元素个数。 |

#### 约束说明

- **不支持源操作数与目的操作数地址重叠。**
- 不支持sharedTmpBuffer与源操作数和目的操作数地址重叠。

- 该接口支持的精度组合如下：

  - half精度组合：srcLocal数据类型=half；scalar数据类型=half；dstLocal数据类型=half；PAR=128
  - float精度组合：srcLocal数据类型=float；scalar数据类型=float；dstLocal数据类型=float；PAR=64
  - mix精度组合：srcLocal数据类型=half；scalar数据类型=half；dstLocal数据类型=float；PAR=64

#### 调用示例

完整的调用样例请参考更多样例。

```
AscendC::TPipe pipe;
AscendC::TQue<AscendC::TPosition::VECCALC, 1> tmpQue;
pipe.InitBuffer(tmpQue, 1, bufferSize);  // bufferSize 通过Host侧tiling参数获取
AscendC::LocalTensor<uint8_t> sharedTmpBuffer = tmpQue.AllocTensor<uint8_t>();
// 输入tensor长度为1024, 算子输入的数据类型为half, 实际计算个数为512
AscendC::Axpy(dstLocal, srcLocal, static_cast<half>(3.0), sharedTmpBuffer, 512);
```

结果示例如下：

```
输入数据(srcLocal): [104.875 107.4375 -62.59375 ...  -242.875 15.8828125]
输出数据(dstLocal): [316.5 324.2 185.8  ...  -726.5 49.66]
```
