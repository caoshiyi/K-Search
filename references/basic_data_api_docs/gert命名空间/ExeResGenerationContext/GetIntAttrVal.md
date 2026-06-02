# GetIntAttrVal

**页面ID:** atlasopapi_07_00706  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00706.html

---

#### 函数功能

获取算子整数类型的属性值。

#### 函数原型

```
bool GetIntAttrVal(const char *attr_name, int64_t &val) const
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| attr_name | 输入 | 属性名。 |
| val | 输出 | 接收获取到的整数值。 |

#### 返回值说明

表示是否成功获取到该属性。

#### 约束说明

无

#### 调用示例

```
bool GetIntAttrVal(ExeResGenerationContext* context) {
  std::string attr_name = "";
  int64_t val; 
  auto result = context->GetStrAttrVal(attr_name.c_str(),val);
  // ...
}
```
