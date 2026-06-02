# SetSkipLocalRankCopy

**页面ID:** atlasascendc_api_07_10043  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_10043.html

---

#### 功能说明

设置本卡的通信算法的计算结果是否输出到目的数据buffer地址。

#### 函数原型

```
uint32_t SetSkipLocalRankCopy(uint8_t skipLocalRankCopy)
```

#### 参数说明

**表1 **参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| skipLocalRankCopy | 输入 | 本卡的通信算法的计算结果是否输出到recvBuf（目的数据buffer地址）。 针对Atlas A2 训练系列产品/Atlas A2 推理系列产品，仅AllGather算法与AlltoAll算法支持配置该参数。uint8_t类型，参数取值如下： - 0：输出本卡通信算法的计算结果（未调用本接口时的默认行为）。- 1：不输出本卡通信算法的计算结果。在无需输出通信结果时，配置参数值为1，此时不会拷贝本卡的通信结果数据，可提升算子性能。例如，在8卡场景下，本卡只取其他卡的部分数据，这时可配置该参数为1。 针对Atlas A3 训练系列产品/Atlas A3 推理系列产品，该参数为预留字段，配置后不生效。 |

#### 返回值说明

- 0表示设置成功。
- 非0表示设置失败。

#### 约束说明

无

#### 调用示例

本接口的调用示例请见调用示例。
