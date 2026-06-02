# SetTiling

**页面ID:** atlasascendc_api_07_0979  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0979.html

---

#### 功能说明

注册Tiling函数。Tiling函数的原型是固定的，接受一个TilingContext作为输入，在此context上可以获取到输入、输出的Shape指针等内容。注册的Tiling函数由框架调用，调用时会传入TilingContext参数。

#### 函数原型

```
OpAICoreDef &SetTiling(gert::OpImplRegisterV2::TilingKernelFunc func)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| Tiling函数。TilingKernelFunc类型定义如下：                                                                                                                           ``` using TilingKernelFunc = UINT32 (*)(TilingContext *); ``` |  |  |

#### 返回值说明

OpAICoreDef算子定义，OpAICoreDef请参考OpAICoreDef。

#### 约束说明

无
