# AddAttr

**页面ID:** atlasascendc_api_07_1019  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_1019.html

---

#### 功能说明

设置算子的属性以及对应值

#### 函数原型

```
ContextBuilder &AddAttr(const std::string& attrName, int64_t attrValue)
ContextBuilder &AddAttr(const std::string& attrName, bool attrValue)
ContextBuilder &AddAttr(const std::string& attrName, const std::string& attrValue)
ContextBuilder &AddAttr(const std::string& attrName, float attrValue)
ContextBuilder &AddAttr(const std::string& attrName, const std::vector<float>& attrValue)
ContextBuilder &AddAttr(const std::string& attrName, const std::vector<bool>& attrValue)
ContextBuilder &AddAttr(const std::string& attrName, const std::vector<int64_t>& attrValue)
ContextBuilder &AddAttr(const std::string& attrName, const std::vector<std::string>& attrValue)
ContextBuilder &AddAttr(const std::string& attrName, const std::vector<std::vector<int64_t>>& attrValue)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| attrName | 输入 | 算子的属性名称 |
| attrValue | 输入 | 算子属性值，支持int64_t、bool、std::string、float、std::vector<float>、std::vector<int64_t>、 std::vector<std::string>、std::vector<bool>、std::vector<std::vector<int64_t>>类型的指定。 |

#### 返回值说明

当前ContextBuilder的对象。

#### 约束说明

AddAttr不支持重复添加同名的属性

#### 调用示例

```
context_ascendc::ContextBuilder builder;
auto builder
    .NodeIoNum(1,1)
    .IrInstanceNum({1})
    .AddAttr("attr_1", 1)
    .AddAttr("attr_2", true)
    .AddAttr("attr_3", "stringValue")
    .AddAttr("attr_4", 1.f)
    .AddAttr("attr_5", {1})
    .AddAttr("attr_6", {false})
    .AddAttr("attr_7", {"stringValue"})
    .AddAttr("attr_8", {1.f})
    .AddAttr("attr_9", {{1, 2}, {3, 4}})
```
