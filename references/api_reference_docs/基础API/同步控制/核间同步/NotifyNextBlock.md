# NotifyNextBlock

**页面ID:** atlasascendc_api_07_0208  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0208.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品            /             Atlas A3 推理系列产品 | x |
| Atlas A2 训练系列产品            /             Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | x |
| Atlas 推理系列产品            AI Core | √ |
| Atlas 推理系列产品            Vector Core | x |
| Atlas 训练系列产品 | x |

#### 功能说明

通过写GM地址，通知下一个核当前核的操作已完成，下一个核可以进行操作。使用接口前，请确保已经调用InitDetermineComputeWorkspace接口，初始化共享内存。

#### 函数原型

```
__aicore__ inline void NotifyNextBlock(GlobalTensor<int32_t>& gmWorkspace, LocalTensor<int32_t>& ubWorkspace)
```

#### 参数说明

**表1 **接口参数说明

| 参数名称 | 输入/输出 | 含义 |
| --- | --- | --- |
| gmWorkspace | 输入 | 临时空间，通过写gmWorkspace通知其他核当前核已执行完成，其他核可以继续往下执行，类型为GlobalTensor。 |
| ubWorkspace | 输入 | 临时空间，用于操作gmWorkspace，类型为LocalTensor。 |

#### 约束说明

- 需要保证每个核调用该接口的次数相同。
- gmWorkspace申请的空间最少要求为：blockNum * 32Bytes；ubWorkspace申请的空间最少要求为：blockNum * 32 + 32Bytes；其中blockNum为调用的核数，可调用GetBlockNum获取。
- 分离模式下，使用该接口进行多核同步时，仅对AIV核生效，WaitPreBlock和NotifyNextBlock之间仅支持插入矢量计算相关指令，对矩阵计算相关指令不生效。
- 使用该接口进行多核控制时，算子调用时指定的逻辑blockDim必须保证不大于实际运行该算子的AI处理器核数，否则框架进行多轮调度时会插入异常同步，导致Kernel“卡死”现象。

#### 调用示例

请参考调用示例。
