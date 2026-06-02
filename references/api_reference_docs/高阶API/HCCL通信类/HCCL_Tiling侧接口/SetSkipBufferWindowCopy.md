# SetSkipBufferWindowCopy

**页面ID:** atlasascendc_api_07_10044  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_10044.html

---

#### 功能说明

设置通信算法获取输入数据的位置。

#### 函数原型

```
uint32_t SetSkipBufferWindowCopy(uint8_t skipBufferWindowCopy)
```

#### 参数说明

**表1 **参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| skipBufferWindowCopy | 输入 | 表示通信算法获取输入数据的位置，uint8_t类型。 针对Atlas A2 训练系列产品/Atlas A2 推理系列产品，参数取值如下： - 0：通信输入不放在windows中。未调用该接口设置通信输入的位置时，默认通信输入不放在windows中。其中windows为其他卡可访问的共享缓冲区。- 1：通信输入不放在windows中，当前该参数取值1与取值0的功能一致。- 2：通信输入放在windows中，仅适用于AllReduce算法、AlltoAll算法。 针对Atlas A3 训练系列产品/Atlas A3 推理系列产品，该参数为预留字段，配置后不生效。 |

#### 返回值说明

- 0表示设置成功。
- 非0表示设置失败。

#### 约束说明

无

#### 调用示例

本接口的调用示例请见调用示例。
