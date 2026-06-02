# SetStrAttrVal

**页面ID:** atlasopapi_07_00705  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00705.html

---

#### 函数功能

设置一个字符串类型的属性值到算子中。

#### 函数原型

```
bool SetStrAttrVal(const char *attr_name, const char *val) const
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| attr_name | 输入 | 属性名。 |
| val | 输入 | 要设置的字符串。 |

#### 返回值说明

表示是否成功设置该属性值。

#### 约束说明

无

#### 调用示例

```
bool SetStrAttrVal(ExeResGenerationContext* context) {
  std::string attr_name = "";
  std::string val = ""; 
  bool result = context->SetStrAttrVal(attr_name.c_str(),val.c_str());
  // ...
}
```
