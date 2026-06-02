# Workspace

**页面ID:** atlasascendc_api_07_1023  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_1023.html

---

#### 功能说明

传入指向gert::ContinuousVector的指针。

#### 函数原型

```
ContextBuilder &Workspace(gert::ContinuousVector *workspace)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| workspace | 输入 | 指向gert::ContinuousVector类的void*指针 |

#### 返回值说明

当前ContextBuilder的对象。

#### 约束说明

由于TilingContext与KernelContext、TilingParseContext内部数据排序不同，Workspace()只支持以调用BuildTilingContext()为前提来使用；其他场景建议用Outputs接口，否则发生未定义行为。

#### 调用示例

```
void AddWorkspaceData(gert::ContinuousVector *ws)
{
    ......
    auto builder = context_ascendc::ContextBuilder()
                                    .Workspace(ws);
                                    .BuildTilingContext();
    ......
}
```
