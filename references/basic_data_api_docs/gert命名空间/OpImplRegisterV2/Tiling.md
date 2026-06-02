# Tiling

**页面ID:** atlasopapi_07_00118  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00118.html

---

#### 函数功能

注册算子的Tiling函数。

用户需要为算子编写一个TilingKernelFunc类型的函数，并使用该接口进行注册。同时可以指定tiling数据的最大长度，缺省值为2048字节。

TilingKernelFunc类型定义如下：

```
using TilingKernelFunc = UINT32 (*)(TilingContext *);
```

#### 函数原型

```
OpImplRegisterV2 &Tiling(TilingKernelFunc tiling_func, size_t max_tiling_data_size = 2048)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| tiling_func | 输入 | 待注册的自定义Tiling函数，类型为TilingKernelFunc。 |
| max_tiling_data_size | 输入 | tiling数据的最大长度，默认值为2048字节。 |

#### 返回值说明

返回算子的OpImplRegisterV2对象，该对象新增注册了Tiling函数tiling_func。

#### 约束说明

无。
