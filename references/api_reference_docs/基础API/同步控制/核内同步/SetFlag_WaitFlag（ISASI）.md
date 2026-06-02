# SetFlag/WaitFlag(ISASI)

**页面ID:** atlasascendc_api_07_0270  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0270.html

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

同一核内不同流水之间的同步指令。具有数据依赖的不同流水指令之间需要插此同步。

#### 函数原型

```
template <HardEvent event>
__aicore__ inline void SetFlag(int32_t eventID)
template <HardEvent event>
__aicore__ inline void WaitFlag(int32_t eventID)
```

#### 参数说明

**表1 **参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| event | 输入 | 模板参数。 同步事件，数据类型为HardEvent。详细内容参考下文中的同步类型说明。 |
| eventID | 输入 | 事件ID。数据类型为int32_t类型。其定义如下： eventID需要通过AllocEventID或者FetchEventID来获取。 Atlas 训练系列产品，数据范围为：0-3 Atlas 推理系列产品AI Core，数据范围为：0-7 Atlas A2 训练系列产品/Atlas A2 推理系列产品，数据范围为：0-7 Atlas A3 训练系列产品/Atlas A3 推理系列产品，数据范围为：0-7 |

同步类型说明如下:

```
enum class HardEvent : uint8_t {
    // 名称（源流水_目标流水），例如MTE2_V，代表PIPE_MTE2为源流水，PIPE_V为目标流水。标识从PIPE_MTE2到PIPE_V的同步，PIPE_V等待PIPE_MTE2。
    MTE2_MTE1
    MTE1_MTE2
    MTE1_M
    M_MTE1
    MTE2_V
    V_MTE2
    MTE3_V
    V_MTE3
    M_V
    V_M
    V_V
    MTE3_MTE1
    MTE1_MTE3
    MTE1_V
    MTE2_M
    M_MTE2
    V_MTE1
    M_FIX
    FIX_M
    MTE3_MTE2
    MTE2_MTE3
    S_V
    V_S
    S_MTE2
    MTE2_S
    S_MTE3
    MTE3_S
    MTE2_FIX
    FIX_MTE2
    FIX_S
    M_S
    FIX_MTE3
}
```

#### 约束说明

- SetFlag/WaitFlag必须成对出现。
- 禁止用户在使用SetFlag和WaitFlag时，自行指定eventID，容易与框架同步事件冲突，导致卡死问题。eventID需要

通过AllocEventID或者FetchEventID来获取。

#### 调用示例

如DataCopy需要等待SetValue执行完成后才能执行，需要插入PIPE_S到PIPE_MTE3的同步。

```
AscendC::GlobalTensor<half> dstGlobal;
AscendC::LocalTensor<half> dstLocal;
dstLocal.SetValue(0, 0);
uint32_t dataSize = 512;
int32_t eventIDSToMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_MTE3));
AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(eventIDSToMTE3);
AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(eventIDSToMTE3);
AscendC::DataCopy(dstGlobal, dstLocal, dataSize);
```
