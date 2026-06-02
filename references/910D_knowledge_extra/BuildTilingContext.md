# BuildTilingContext<a name="ZH-CN_TOPIC_0000002523303572"></a>

## 功能说明<a name="zh-cn_topic_0000001867289945_zh-cn_topic_0000001389787297_section36583473819"></a>

构造KernelRunContextHolder的对象，该对象可通过GetContext接口获取TilingContext类型的对象。

## 函数原型<a name="zh-cn_topic_0000001867289945_zh-cn_topic_0000001389787297_section13230182415108"></a>

```
std::shared_ptr<KernelRunContextHolder> BuildTilingContext()
```

## 参数说明<a name="zh-cn_topic_0000001867289945_zh-cn_topic_0000001389787297_section75395119104"></a>

无

## 返回值说明<a name="zh-cn_topic_0000001867289945_zh-cn_topic_0000001389787297_section25791320141317"></a>

指向KernelRunContextHolder的共享指针。

## 约束说明<a name="zh-cn_topic_0000001867289945_zh-cn_topic_0000001389787297_section19165124931511"></a>

无

## 调用示例<a name="zh-cn_topic_0000001867289945_zh-cn_topic_0000001389787297_section320753512363"></a>

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

