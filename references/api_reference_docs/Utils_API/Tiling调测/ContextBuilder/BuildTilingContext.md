# BuildTilingContext

**页面ID:** atlasascendc_api_07_1024  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_1024.html

---

#### 功能说明

构造KernelRunContextHolder的对象，该对象可通过GetContext接口获取TilingContext类型的对象。

#### 函数原型

```
std::shared_ptr<KernelRunContextHolder> BuildTilingContext()
```

#### 参数说明

无

#### 返回值说明

指向KernelRunContextHolder的共享指针。

#### 约束说明

无

#### 调用示例

```
auto tilingContextHolder = context_ascendc::ContextBuilder().
  .SetOpNameType(...,...)
  .NodeIoNum(...)
  .IrInstanceNum(...)
  .AddInputTd(...)
  .AddOutputTd(...)
  .AddAttr(...)
  .BuildTilingContext(...);
gert::TilingContext *tilingContext = tilingContextHolder->GetContext<gert::TilingContext>();
```
