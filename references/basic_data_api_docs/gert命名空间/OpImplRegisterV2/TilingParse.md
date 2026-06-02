# TilingParse

**页面ID:** atlasopapi_07_00120  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00120.html

---

#### 函数功能

注册算子的TilingParse函数，用于解析算子编译阶段生成的算子信息JSON文件，在注册时需要注册算子自行指定数据类型T，该数据类型用于保存解析后的算子信息。

用户需要为算子编写一个KernelFunc类型或者TilingParseFunc类型的函数，并使用下列对应的接口进行注册。

KernelFunc类型定义如下：

```
using KernelFunc = UINT32 (*)(KernelContext *context);
```

TilingParseFunc类型定义如下：

```
using TilingParseFunc = UINT32 (*)(TilingParseContext *context);
```

#### 函数原型

```
template<typename T>
OpImplRegisterV2 &TilingParse(KernelFunc const tiling_parse_func)
template<typename T>
OpImplRegisterV2 &TilingParse(TilingParseFunc const tiling_parse_func)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| tiling_parse_func | 输入 | 待注册的TilingParse函数，类型支持2种：KernelFunc、TilingParseFunc。 |

#### 返回值说明

返回算子的OpImplRegisterV2对象，该对象新增注册了TilingParse函数。

#### 约束说明

无。
