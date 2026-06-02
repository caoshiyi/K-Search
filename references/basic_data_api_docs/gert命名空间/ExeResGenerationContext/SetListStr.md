# SetListStr

**页面ID:** atlasopapi_07_00703  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00703.html

---

#### 函数功能

设置字符串列表类型的属性值到算子中。

#### 函数原型

```
ge::graphStatus SetListStr(const std::string &attr_name, const std::vector<std::string> &list) const
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| attr_name | 输入 | 属性名。 |
| list | 输入 | 要设置的字符串列表。 |

#### 返回值说明

返回graphStatus状态码。

#### 约束说明

无

#### 调用示例

```
ge::graphStatus SetListStr(ExeResGenerationContext* context) {
  std::string attr_name = "";
  std::vector<std::string> list;  
  ge::graphStatus status = context->SetListStr(attr_name ,list);
  // ...
}
```
