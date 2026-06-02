# CompileInfo

**页面ID:** atlasascendc_api_07_1020  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_1020.html

---

#### 功能说明

将指向CompileInfo的指针传入TilingContext

#### 函数原型

```
ContextBuilder &CompileInfo(void *compileInfo)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| compileInfo | 输入 | 指向CompileInfo的void指针 |

#### 返回值说明

当前ContextBuilder的对象。

#### 约束说明

由于TilingContext与KernelContext、TilingParseContext内部数据排序不同，CompileInfo()只支持以调用BuildTilingContext()为前提来使用；其他场景建议用Outputs接口，否则发生未定义行为。

#### 调用示例

```
void AddCompileInfo(TilingParseContext *tilingParseContext)
{
    ......
    void *compilerInfo = *tilingParseContext->GetOutputPointer<void **>(0);
    auto kernelContextHolder = context_ascendc::ContextBuilder()
                                    ...... // 增加算子输入输出接口的调用
                                    .CompileInfo(compileInfo)
                                    .BuildTilingContext();
    ......
}
```
