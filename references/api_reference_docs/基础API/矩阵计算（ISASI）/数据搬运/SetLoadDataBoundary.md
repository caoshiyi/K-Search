# SetLoadDataBoundary

**页面ID:** atlasascendc_api_07_0246  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0246.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | x |
| Atlas 推理系列产品AI Core | √ |
| Atlas 推理系列产品Vector Core | x |
| Atlas 训练系列产品 | x |

#### 功能说明

设置Load3D时A1/B1边界值。

如果Load3D指令在处理源操作数时，源操作数在A1/B1上的地址超出设置的边界，则会从A1/B1起始地址开始读取数据。

#### 函数原型

```
__aicore__ inline void SetLoadDataBoundary(uint32_t boundaryValue)
```

#### 参数说明

**表1 **参数说明

| 参数名称 | 输入/输出 | 含义 |
| --- | --- | --- |
| boundaryValue | 输入 | 边界值。 Load3Dv1指令：单位是32字节。 Load3Dv2指令：单位是字节。 |

#### 约束说明

- 用于Load3Dv1时，boundaryValue的最小值是16（单位：32字节）；用于Load3Dv2时，boundaryValue的最小值是1024（单位：字节）。
- 如果使用SetLoadDataBoundary接口设置了边界值，配合Load3D指令使用时，Load3D指令的A1/B1初始地址要在设置的边界内。
- 如果boundaryValue设置为0，则表示无边界，可使用整个A1/B1。

#### 调用示例

参考调用示例。
