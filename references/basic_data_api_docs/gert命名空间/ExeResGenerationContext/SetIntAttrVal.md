# SetIntAttrVal

**页面ID:** atlasopapi_07_00707  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00707.html

---

#### 函数功能

设置一个整数类型的属性值到算子中。

#### 函数原型

```
bool SetIntAttrVal(const char *attr_name, const int64_t val) const
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| attr_name | 输入 | 属性名。 |
| val | 输入 | 要设置的整数值。 |

#### 返回值说明

表示是否成功设置该属性值。

#### 约束说明

无

#### 调用示例

```
bool SetIntAttrVal(ExeResGenerationContext* context) {
  std::string attr_name = "";
  int64_t val = 6; 
  auto result = context->SetIntAttrVal(attr_name.c_str(),val);
  // ...
}
```
