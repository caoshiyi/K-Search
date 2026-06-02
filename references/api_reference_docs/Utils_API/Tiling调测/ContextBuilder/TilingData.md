# TilingData

**页面ID:** atlasascendc_api_07_1022  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_1022.html

---

#### 功能说明

传入指向TilingData的void* 指针

#### 函数原型

```
ContextBuilder &TilingData(void *tilingData)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| tilingData | 输入 | 指向TilingData类的void*指针 |

#### 返回值说明

当前ContextBuilder的对象。

#### 约束说明

由于TilingContext与KernelContext、TilingParseContext内部数据排序不同，TilingData()只支持以调用BuildTilingContext()为前提来使用；其他场景建议用Outputs接口，否则发生未定义行为。

#### 调用示例

```
void AddTilingData(void *tilingData)
{
    ......
    auto kernelContextHolder = context_ascendc::ContextBuilder()
                                    .TilingData(tilingData)
                                    .BuildTilingContext();
    ......
}
```
