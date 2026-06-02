# FetchEventID

**页面ID:** atlasascendc_api_07_0116  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0116.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | x |
| Atlas 推理系列产品AI Core | √ |
| Atlas 推理系列产品Vector Core | x |
| Atlas 训练系列产品 | √ |

#### 功能说明

根据HardEvent（硬件类型的同步事件）获取相应可用的TEventID，此接口不会申请TEventID，仅提供可用的TEventID。

#### 函数原型

```
template <HardEvent evt>
__aicore__ inline TEventID FetchEventID()
__aicore__ inline TEventID FetchEventID(HardEvent evt)
```

#### 参数说明

| 参数名 | 输入/输出 | 含义 |
| --- | --- | --- |
| evt | 输入 | HardEvent类型，硬件同步类型。 该类型的具体说明请参考SetFlag/WaitFlag(ISASI)中同步类型的说明。 |

#### 约束说明

相比于AllocEventID，FetchEventID适用于临时使用ID的场景，获取ID后，不会对ID进行占用。在一些复杂的使用场景下，需要开发者自行保证使用正确。比如相同流水连续调用SetFlag/WaitFlag，如果两次传入的ID都是使用FetchEventID获取的，因为两者ID相同会出现程序卡死等未定义行为，这时推荐用户使用AllocEventID。

#### 返回值说明

TEventID

#### 调用示例

```
AscendC::TEventID eventIdVToS = GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_S); //需要插scalar等vector的同步，申请对应的HardEvent的ID
AscendC::SetFlag<AscendC::HardEvent::V_S>(eventIdVToS);
AscendC::WaitFlag<AscendC::HardEvent::V_S>(eventIdVToS);
```
