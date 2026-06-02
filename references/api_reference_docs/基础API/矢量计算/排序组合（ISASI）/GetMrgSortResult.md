# GetMrgSortResult

**页面ID:** atlasascendc_api_07_0233  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0233.html

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

获取MrgSort已经处理过的队列里的Region Proposal个数，并依次存储在四个出参中。

本接口和MrgSort相关指令的配合关系如下：

- 配合MrgSort4指令使用，获取MrgSort4指令处理过的队列里的Region Proposal个数。使用时，需要将MrgSort4中的MrgSort4Info.ifExhaustedSuspension参数配置为true，该配置模式下某条队列耗尽后，MrgSort4指令即停止。

以上说明适用于如下型号：

Atlas 推理系列产品AI Core

- 配合MrgSort指令使用，获取MrgSort指令处理过的队列里的Region Proposal个数。使用时，需要将MrgSort中的MrgSort4Info.ifExhaustedSuspension参数配置为true，该配置模式下某条队列耗尽后，MrgSort指令即停止。

以上说明适用于如下型号：

Atlas A3 训练系列产品/Atlas A3 推理系列产品

Atlas A2 训练系列产品/Atlas A2 推理系列产品

Atlas 200I/500 A2 推理产品

#### 函数原型

```
__aicore__ inline void GetMrgSortResult(uint16_t &mrgSortList1, uint16_t &mrgSortList2, uint16_t &mrgSortList3, uint16_t &mrgSortList4)
```

#### 参数说明

**表1 **参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| mrgSortList1 | 输出 | 类型为uint16_t，表示MrgSort第一个队列里已经处理过的Region Proposal个数。 |
| mrgSortList2 | 输出 | 类型为uint16_t，表示MrgSort第二个队列里已经处理过的Region Proposal个数。 |
| mrgSortList3 | 输出 | 类型为uint16_t，表示MrgSort第三个队列里已经处理过的Region Proposal个数。 |
| mrgSortList4 | 输出 | 类型为uint16_t，表示MrgSort第四个队列里已经处理过的Region Proposal个数。 |

#### 约束说明

无

#### 调用示例

- 配合MrgSort指令使用示例。

```
AscendC::LocalTensor<float> dstLocal;
AscendC::LocalTensor<float> workLocal;
AscendC::LocalTensor<float> src0Local;
AscendC::LocalTensor<uint32_t> src1Local;

AscendC::Sort32(workLocal, src0Local, src1Local, 1);

uint16_t elementLengths[4] = { 0 };
uint32_t sortedNum[4] = { 0 };
elementLengths[0] = 32;
elementLengths[1] = 32;
elementLengths[2] = 32;
elementLengths[3] = 32;
uint16_t validBit = 0b1111;

AscendC::MrgSortSrcList<float> srcList;
srcList.src1 = workLocal[0];
srcList.src2 = workLocal[32 * 1 * 2];
srcList.src3 = workLocal[32 * 2 * 2];
srcList.src4 = workLocal[32 * 3 * 2];

AscendC::MrgSort4Info mrgSortInfo(elementLengths, true, validBit, 1);
AscendC::MrgSort(dstLocal, srcList, mrgSortInfo);

uint16_t mrgRes1 = 0;
uint16_t mrgRes2 = 0;
uint16_t mrgRes3 = 0;
uint16_t mrgRes4 = 0;
AscendC::GetMrgSortResult(mrgRes1, mrgRes2, mrgRes3, mrgRes4);
```

- 配合MrgSort4指令使用示例。

```
AscendC::LocalTensor<float> dstLocal;
AscendC::LocalTensor<float> workLocal;
AscendC::LocalTensor<float> src0Local;

AscendC::RpSort16(workLocal, src0Local, 1);

uint16_t elementLengths[4] = { 0 };
uint32_t sortedNum[4] = { 0 };
elementLengths[0] = 32;
elementLengths[1] = 32;
elementLengths[2] = 32;
elementLengths[3] = 32;
uint16_t validBit = 0b1111;

AscendC::MrgSortSrcList<float> srcList;
srcList.src1 = workLocal[0];
srcList.src2 = workLocal[32 * 1 * 2];
srcList.src3 = workLocal[32 * 2 * 2];
srcList.src4 = workLocal[32 * 3 * 2];

AscendC::MrgSort4Info mrgSortInfo(elementLengths, true, validBit, 1);
AscendC::MrgSort4(dstLocal, srcList, mrgSortInfo);

uint16_t mrgRes1 = 0;
uint16_t mrgRes2 = 0;
uint16_t mrgRes3 = 0;
uint16_t mrgRes4 = 0;
AscendC::GetMrgSortResult(mrgRes1, mrgRes2, mrgRes3, mrgRes4);
```
