# GetWindowsOutAddr

**页面ID:** atlasascendc_api_07_0883  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0883.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | x |
| Atlas 推理系列产品AI Core | x |
| Atlas 推理系列产品Vector Core | x |
| Atlas 训练系列产品 | x |

#### 功能说明

获取卡间通信数据WindowsOut起始地址，可用来直接作为计算的输入输出地址，减少拷贝。该接口默认在所有核上工作，用户也可以在调用前通过GetBlockIdx指定其在某一个核上运行。

#### 函数原型

```
__aicore__ inline GM_ADDR GetWindowsOutAddr(uint32_t rankId)
```

#### 参数说明

**表1 **接口参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| rankId | 输入 | 待查询的卡的Id。 |

#### 返回值说明

返回对应卡的卡间通信数据WindowsOut起始地址。当rankId非法时，返回nullptr。

#### 约束说明

无

#### 调用示例

请参见GetWindowsInAddr的调用示例。
