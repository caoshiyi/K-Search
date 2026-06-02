# QueueBarrier

**页面ID:** atlasascendc_api_07_10199  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_10199.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | x |
| Atlas 200I/500 A2 推理产品 | x |
| Atlas 推理系列产品AI Core | x |
| Atlas 推理系列产品Vector Core | x |
| Atlas 训练系列产品 | x |

#### 功能说明

阻塞服务端上指定队列的BatchWrite通信任务，直到指定范围内所有队列上的任务完成执行，从而实现指定范围内队列的同步。

#### 函数原型

```
template <ScopeType type = ScopeType::ALL>
__aicore__ inline void QueueBarrier(uint16_t queueID)
```

#### 参数说明

**表1 **模板参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| 表示阻塞服务端的通信任务范围。类型为ScopeType，默认值为ScopeType::ALL。当前参数仅支持取值为ScopeType::ALL。 ScopeType的定义如下： ``` enum class ScopeType: uint8_t {     ALL, // 阻塞所有队列上的通信任务     QUEUE, // 暂不支持     BLOCK, // 暂不支持     INVALID_TYPE // 暂不支持 }; ``` |  |  |

**表2 **接口参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| queueID | 输入 | 表示队列ID。 |

#### 约束说明

无

#### 调用示例

请参见BatchWrite的调用示例。
