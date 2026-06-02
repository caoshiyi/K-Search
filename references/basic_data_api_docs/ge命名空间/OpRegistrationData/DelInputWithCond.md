# DelInputWithCond

**页面ID:** atlasopapi_07_00396  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00396.html

---

#### 函数功能

根据算子属性，删除算子指定输入边。

#### 函数原型

> **注意:** 

数据类型为string的接口后续版本会废弃，建议使用数据类型为非string的接口。

```
OpRegistrationData &DelInputWithCond(int32_t inputIdx, const std::string &attrName, bool attrValue)
OpRegistrationData &DelInputWithCond(int32_t input_idx, const char_t *attr_name, bool attr_value)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| input_idx/inputIdx | 输入 | 需要删除的输入边编号。 |
| attr_name/attrName | 输入 | 属性名字。 |
| attr_value/attrValue | 输入 | 属性的值。 |

#### 约束说明

无。
