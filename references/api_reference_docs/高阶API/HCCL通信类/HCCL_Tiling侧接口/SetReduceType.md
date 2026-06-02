# SetReduceType

**页面ID:** atlasascendc_api_07_10041  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_10041.html

---

#### 功能说明

设置Reduce操作类型，仅对有归约操作的通信任务生效。

#### 函数原型

```
uint32_t SetReduceType(uint32_t reduceType, uint8_t dstDataType = 0, uint8_t srcDataType = 0)
```

#### 参数说明

**表1 **参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| reduceType | 输入 | 归约操作类型，仅对有归约操作的通信任务生效。uint32_t类型，取值详见表2参数说明。 |
| dstDataType | 输入 | 通信任务中输出数据的数据类型。uint8_t类型，该参数的取值范围请参考表1。 Atlas A3 训练系列产品/Atlas A3 推理系列产品，该参数暂不支持，配置后不生效。 Atlas A2 训练系列产品/Atlas A2 推理系列产品，该参数暂不支持，配置后不生效。 |
| srcDataType | 输入 | 通信任务中输入数据的数据类型。uint8_t类型，该参数的取值范围请参考表1。 Atlas A3 训练系列产品/Atlas A3 推理系列产品，该参数暂不支持，配置后不生效。 针对Atlas A2 训练系列产品/Atlas A2 推理系列产品，该参数暂不支持，配置后不生效。 |

#### 返回值说明

- 0表示设置成功。
- 非0表示设置失败。

#### 约束说明

无

#### 调用示例

本接口的调用示例请见调用示例。
