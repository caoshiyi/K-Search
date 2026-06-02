# SetInferShape

**页面ID:** atlasascendc_api_07_0950  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0950.html

---

#### 功能说明

使用图模式时，需要调用该接口注册Shape推导函数。

#### 函数原型

```
OpDef &SetInferShape(gert::OpImplRegisterV2::InferShapeKernelFunc func)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| Shape推导函数。InferShapeKernelFunc类型定义如下，入参类型参考InferShapeContext：                                                                                                                           ``` using InferShapeKernelFunc = UINT32 (*)(InferShapeContext *); ``` |  |  |

#### 返回值说明

OpDef算子定义，OpDef请参考OpDef。

#### 约束说明

无
