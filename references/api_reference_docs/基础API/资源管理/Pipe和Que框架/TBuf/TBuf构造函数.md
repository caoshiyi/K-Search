# TBuf构造函数

**页面ID:** atlasascendc_api_07_0162  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0162.html

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

创建TBuf对象时，初始化数据成员。

#### 函数原型

```
template <TPosition pos = TPosition::LCM>
__aicore__ inline TBuf();
```

#### 参数说明

**表1 **模板参数说明

| 参数名称 | 含义 |
| --- | --- |
| pos | TBuf所在的逻辑位置，取值为VECCALC。关于TPosition的具体介绍请参考TPosition。 |

#### 约束说明

无。
