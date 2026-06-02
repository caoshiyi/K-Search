# AutoMappingFn

**页面ID:** atlasopapi_07_00408  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00408.html

---

#### 函数功能

自动映射回调函数。

#### 函数原型

```
Status AutoMappingFn(const google::protobuf::Message *op_src, ge::Operator &op)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| op_src | 输入 | 转换前原始模型中的算子，包含原始模型中算子的属性。 |
| op | 输入 | 适配昇腾AI处理器的算子。          关于Operator类，请参见Operator。 |

#### 约束说明

若原始TensorFlow算子与适配昇腾AI处理器的算子属性无法一一映射，AutoMappingFn函数无法应用于回调函数ParseParamsByOperatorFn中，此种场景下，请在回调函数ParseParamsByOperatorFn中使用AutoMappingByOpFn接口进行可以映射成功的属性的自动解析，使用示例请参见调用示例。
