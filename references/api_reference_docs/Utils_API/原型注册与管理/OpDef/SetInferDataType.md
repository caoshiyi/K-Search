# SetInferDataType

**页面ID:** atlasascendc_api_07_0952  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0952.html

---

#### 功能说明

使用图模式时，需要调用该接口注册DataType推导函数。

#### 函数原型

```
OpDef &SetInferDataType(gert::OpImplRegisterV2::InferDataTypeKernelFunc func)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| DataType推导函数。**InferDataTypeKernelFunc**类型定义如下，入参类型参考InferDataTypeContext：                                                                                                                           ``` using InferDataTypeKernelFunc = UINT32 (*)(InferDataTypeContext *); ``` |  |  |

#### 返回值说明

OpDef算子定义，OpDef请参考OpDef。

#### 约束说明

无
