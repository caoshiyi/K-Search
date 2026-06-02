# GenSimplifiedKey

**页面ID:** atlasopapi_07_00123  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00123.html

---

#### 函数功能

注册算子的GenSimplifiedKey函数，以提供一个更加快速的二进制匹配key值。

用户需要为算子编写一个GenSimplifiedKey类型的函数，并使用该接口进行注册。

GenSimplifiedKey类型定义如下：

```
using GenSimplifiedKeyKernelFunc = UINT32 (*)(TilingContext *, ge::char_t *);
```

GenSimplifiedKey函数接受一个TilingContext类型参数和ge::char_t类型参数作为输入，通过该输入算子可自定义simplified key生成逻辑。

#### 函数原型

```
OpImplRegisterV2 &GenSimplifiedKey(GenSimplifiedKeyKernelFunc gen_simplifiedkey_func)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| 要注册的自定义GenSimplifiedKey函数，类型为GenSimplifiedKeyKernelFunc。 GenSimplifiedKeyKernelFunc类型定义如下： ``` using GenSimplifiedKeyKernelFunc = UINT32 (*)(TilingContext *, ge::char_t *); ``` |  |  |

#### 返回值说明

返回算子的OpImplRegisterV2对象，该对象新增注册了生成二进制简化匹配key函数。

#### 约束说明

无。
