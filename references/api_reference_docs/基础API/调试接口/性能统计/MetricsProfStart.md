# MetricsProfStart

**页面ID:** atlasascendc_api_07_1214  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_1214.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品            /             Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品            /             Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | x |
| Atlas 推理系列产品            AI Core | x |
| Atlas 推理系列产品            Vector Core | x |
| Atlas 训练系列产品 | x |

#### 功能说明

用于设置性能数据采集信号启动，和MetricsProfStop配合使用。使用msProf工具进行算子上板调优时，可在kernel侧代码段前后分别调用MetricsProfStart和MetricsProfStop来指定需要调优的代码段范围。

#### 函数原型

```
__aicore__ inline void MetricsProfStart()
```

#### 参数说明

无

#### 约束说明

无

#### 调用示例

```
MetricsProfStart();
```
