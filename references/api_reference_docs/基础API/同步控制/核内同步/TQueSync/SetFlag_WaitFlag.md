# SetFlag/WaitFlag

**页面ID:** atlasascendc_api_07_0181  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0181.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品AI Core | √ |
| Atlas 推理系列产品Vector Core | x |
| Atlas 训练系列产品 | √ |

#### 功能说明

同一核内不同流水线之间的同步指令，具有数据依赖的不同流水指令之间需要插入此同步。

- SetFlag：当前序指令的所有读写操作都完成之后，当前指令开始执行，并将硬件中的对应标志位设置为1。
- WaitFlag：当执行到该指令时，如果发现对应标志位为0，该队列的后续指令将一直被阻塞；如果发现对应标志位为1，则将对应标志位设置为0，同时后续指令开始执行。

#### 函数原型

```
__aicore__ inline void SetFlag(TEventID id)
__aicore__ inline void WaitFlag(TEventID id)
```

#### 参数说明

**表1 **参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| 事件ID，由用户自己指定，推荐通过AllocEventID或者FetchEventID来获取。定义如下： ``` using TEventID = int8_t; ```  Atlas 训练系列产品，数据范围为：0-3 Atlas 推理系列产品AI Core，数据范围为：0-7 Atlas A2 训练系列产品/Atlas A2 推理系列产品，数据范围为：0-7 Atlas A3 训练系列产品/Atlas A3 推理系列产品，数据范围为：0-7 Atlas 200I/500 A2 推理产品，数据范围为：0-7 |  |  |

#### 约束说明

SetFlag/WaitFlag必须成对出现。

#### 调用示例

如DataCopy需要等待SetValue执行完成后才能执行，需要插入PIPE_S到PIPE_MTE3的同步。

```
AscendC::GlobalTensor<half> dstGlobal;
AscendC::LocalTensor<half> dstLocal;
dstLocal.SetValue(0, 0);
AscendC::TQueSync<PIPE_S, PIPE_MTE3> sync;
sync.SetFlag(0);
sync.WaitFlag(0);
AscendC::DataCopy(dstGlobal, dstLocal, dataSize);
```
