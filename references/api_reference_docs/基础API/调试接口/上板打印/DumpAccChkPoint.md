# DumpAccChkPoint

**页面ID:** atlasascendc_api_07_0195  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0195.html

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

基于算子工程开发的算子，可以使用该接口Dump指定Tensor的内容。同时支持打印自定义的附加信息（仅支持uint32_t数据类型的信息），比如打印当前行号等。区别于DumpTensor，使用该接口可以支持指定偏移位置的Tensor打印。

在算子kernel侧实现代码中需要打印偏移后Tensor数据的地方调用DumpAccChkPoint接口打印相关内容。样例如下：

```
AscendC::DumpAccChkPoint(srcLocal, 5, 32, dataLen);
```

> **注意:** 

DumpAccChkPoint接口打印功能会对算子实际运行的性能带来一定影响，通常在调测阶段使用。开发者可以按需通过设置ASCENDC_DUMP=0来关闭打印功能。

#### 函数原型

```
template <typename T>
__aicore__ inline void DumpAccChkPoint(const LocalTensor<T> &tensor, uint32_t index, uint32_t countOff, uint32_t dumpSize)
template <typename T>
__aicore__ inline void DumpAccChkPoint(const GlobalTensor<T> &tensor, uint32_t index, uint32_t countOff, uint32_t dumpSize)
```

#### 参数说明

**表1 **模板参数说明

| 参数名 | 描述 |
| --- | --- |
| T | 需要dump的Tensor的数据类型。 Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持的数据类型为：bool、uint8_t、int8_t、int16_t、uint16_t、int32_t、uint32_t、int64_t、uint64_t、float、half、bfloat16_t。 Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持的数据类型为：bool、uint8_t、int8_t、int16_t、uint16_t、int32_t、uint32_t、int64_t、uint64_t、float、half、bfloat16_t。 Atlas 推理系列产品AI Core，支持的数据类型为：bool、uint8_t、int8_t、int16_t、uint16_t、int32_t、uint32_t、int64_t、uint64_t、float、half。 |

**表2 **参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| tensor | 输入 | 需要dump的Tensor。 待dump的tensor位于Unified Buffer/L1 Buffer/L0C Buffer时使用LocalTensor类型的tensor参数输入。 待dump的tensor位于Global Memory时使用GlobalTensor类型的tensor参数输入。 |
| index | 输入 | 用户自定义附加信息（行号或其他自定义数字）。 |
| countOff | 输入 | 偏移元素个数。偏移后的Tensor地址需要满足所在物理位置的对齐约束，具体参考通用说明和约束。 |
| dumpSize | 输入 | 需要dump的元素个数。 |

#### 约束说明

- 该功能仅用于NPU上板调试。
- 暂不支持算子入图场景的打印。

- 当前仅支持打印存储位置为Unified Buffer/L1 Buffer/L0C Buffer/Global Memory的Tensor信息。

- 单次调用DumpTensor打印的数据总量不可超过1MB（还包括少量框架需要的头尾信息，通常可忽略）。使用时应注意，如果超出这个限制，则数据不会被打印。
- 在计算数据量时，若Dump的总长度未对齐，需要考虑padding数据的影响。当进行非对齐Dump时，如果实际Dump的元素长度不满足32字节对齐，系统会在其末尾自动补充一定数量的padding数据，以满足对齐要求。例如，Tensor1中用户需要Dump的元素长度为30字节，系统会在其后添加2字节的padding，使总长度对齐到32字节。但在实际解析时，仍只解析原始的30字节数据，padding部分不会被使用。
- 使用自定义算子工程进行算子开发时，接口的打印信息和上文描述有些差异：

Dump时，每个block核的dump信息前会增加对应信息头DumpHead，用于记录核号和资源使用信息；每次Dump的Tensor数据前也会添加信息头DumpTensorHead，用于记录Tensor的相关信息。如下图所示，展示了多核打印场景下的打印信息结构。

<!-- img2text -->
```text
block0
┌──────────┬────────────────┬────────┬────────────────┬────────┬────────────────┬─────┐
│ DumpHead │ DumpTensorHead │ Tensor1│ DumpTensorHead │ Tensor2│ DumpTensorHead │ ... │
└──────────┴────────────────┴────────┴────────────────┴────────┴────────────────┴─────┘

block1
┌──────────┬────────────────┬────────┬────────────────┬────────┬────────────────┬─────┐
│ DumpHead │ DumpTensorHead │ Tensor1│ DumpTensorHead │ Tensor2│ DumpTensorHead │ ... │
└──────────┴────────────────┴────────┴────────────────┴────────┴────────────────┴─────┘

                                  ...

blockn
┌──────────┬────────────────┬────────┬────────────────┬────────┬────────────────┬─────┐
│ DumpHead │ DumpTensorHead │ Tensor1│ DumpTensorHead │ Tensor2│ DumpTensorHead │ ... │
└──────────┴────────────────┴────────┴────────────────┴────────┴────────────────┴─────┘
```

**DumpHead的具体信息如下：**

  - opType：当前运行的算子类型；
  - CoreType：当前运行的核的类型；
  - block dim：开发者设置的算子执行核数；
  - total_block_num：参与dump的核数；
  - block_remain_len：当前核剩余可用的dump的空间；
  - block_initial_space：当前核初始分配的dump空间；
  - rsv：保留字段；
  - magic：内存校验魔术字。

DumpHead打印时，除了上述打印还会自动打印当前所运行核的类型及对应的该类型下的核索引，如：AIV-0。

**DumpTensorHead的具体信息如下：**

  - desc：用户自定义附加信息；
  - addr：Tensor的地址；
  - data_type：Tensor的数据类型；
  - position：表示Tensor所在的物理存储位置，当前仅支持Unified Buffer/L1 Buffer/L0C Buffer/Global Memory。
  - dump_size：表示用户需要dump的元素个数。

DumpAccChkPoint打印结果的最前面会自动打印CANN_VERSION_STR值与CANN_TIMESTAMP值。其中，CANN_VERSION_STR与CANN_TIMESTAMP为宏定义，CANN_VERSION_STR代表CANN软件包的版本号信息，形式为字符串，CANN_TIMESTAMP为CANN软件包发布时的时间戳，形式为数值（uint64_t）。开发者也可在代码中直接使用这两个宏。

打印结果的样例如下：

```
opType=AddCustom, DumpHead: AIV-0, CoreType=AIV, block dim=8, total_block_num=8, block_remain_len=1046912, block_initial_space=1048576, rsv=0, magic=5aa5bccd 
CANN Version: XX.XX,TimeStamp: XXXXXX
DumpTensor: desc=5, addr=40, data_type=float16, position=UB, dump_size=32
[16.000000, 22.000000, 2.000000, 3.000000, 58.000000, 62.000000, 33.000000, 74.000000, 51.000000, 69.000000, 61.000000, 9.000000, 53.000000, 35.000000, 14.000000, 43.000000, 20.000000, 43.000000, 92.000000, 84.000000, 9.000000, 6.000000, 78.000000, 53.000000, 52.000000, 33.000000, 51.000000, 61.000000, 92.000000, 45.000000, 39.000000,34.000000]
...
DumpTensor: desc=5, addr=140, data_type=float16, position=UB, dump_size=32
[41.000000, 91.000000, 12.000000, 32.000000, 28.000000, 49.000000, 2.000000, 75.000000, 11.000000, 32.000000, 17.000000, 31.000000, 70.000000, 38.000000, 76.000000, 87.000000, 61.000000, 8.000000, 55.000000, 70.000000, 17.000000, 37.000000, 35.000000, 58.000000, 94.000000, 31.000000, 50.000000, 29.000000, 13.000000, 37.000000, 79.000000,29.000000]
```

该接口使用Dump功能，一个算子所有使用Dump功能的接口在每个核上Dump的数据总量不可超过1M。请开发者自行控制待打印的内容数据量，超出则不会打印。

#### 调用示例

```
AscendC::DumpAccChkPoint(srcLocal, 7, 32, 128);
```
