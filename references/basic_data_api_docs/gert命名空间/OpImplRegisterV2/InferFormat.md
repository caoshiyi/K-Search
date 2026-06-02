# InferFormat

**页面ID:** atlasopapi_07_00588  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00588.html

---

#### 函数功能

注册算子的InferFormat函数。

对于部分格式敏感的Cube算子，使用更适应底层硬件的内部格式，可以带来较大的性能收益，所以开发者需要实现InferFormat函数并注册。

#### 函数原型

```
OpImplRegisterV2 &InferFormat(InferFormatFunc infer_format_func)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| infer_format_func | 输入 | 要注册的自定义InferFormat函数，类型为InferFormatFunc。 InferFormatFunc类型定义如下： ``` using InferFormatFunc = UINT32 (*)(InferFormatContext *context); ``` |

#### 返回值说明

返回算子的OpImplRegisterV2对象，该对象新增注册了InferFormat函数infer_format_func。

#### 约束说明

无。
