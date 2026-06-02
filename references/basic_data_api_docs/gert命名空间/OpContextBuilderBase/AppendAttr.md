# AppendAttr

**页面ID:** atlasopapi_07_00605  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00605.html

---

#### 函数功能

追加算子IR原型的属性信息，下标从0开始，用于构造各子类Context的基类ExtendedKernelContext中的ExtendedInfo信息。

构造完成后，通过Context的基类接口GetAttr获取到的RuntimeAttrs中属性的顺序与构造时追加的顺序一致。

例如：

```
bool attr0 = true;
int64_t attr1 = 1;
vector<int64_t> attr2 = {1, 2, 3, 4};
context_builder.AppendAttr(attr0).AppendAttr(attr1).AppendAttr(attr2);
```

Build构造完成后结果如下：

```
ctx->GetAttrs()->GetBool(0) -> attr0,
ctx->GetAttrs()->GetInt(1) -> attr1,
ctx->GetAttrs()->GetListInt(2) -> attr2
```

#### 函数原型

```
T &AppendAttr(bool attr)
T &AppendAttr(int64_t attr)
T &AppendAttr(float attr)
T &AppendAttr(const ge::AscendString &attr)
T &AppendAttr(const std::vector<bool> &attr)
T &AppendAttr(const std::vector<int64_t> &attr)
T &AppendAttr(const std::vector<float> &attr)
T &AppendAttr(const std::vector<ge::AscendString> &attr)
T &AppendAttr(const std::vector<std::vector<int64_t>> &attr)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| attr | 输入 | 属性值，当前仅支持以下类型：bool、int64_t、float、AscendString、std::vector<bool>、std::vector<int64_t>、std::vector<float>、std::vector<AscendString>、std::vector<std::vector<int64_t>>。 |

#### 返回值说明

返回子类对象T类型的引用，用于支持子类链式调用。

#### 约束说明

无。
