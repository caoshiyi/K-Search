# DelInputWithOriginalType

**页面ID:** atlasopapi_07_00397  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00397.html

---

#### 函数功能

根据算子类型，删除算子指定输入边。

#### 函数原型

> **注意:** 

数据类型为string的接口后续版本会废弃，建议使用数据类型为非string的接口。

```
OpRegistrationData &DelInputWithOriginalType(int32_t input_idx, const std::string &ori_type)
OpRegistrationData &DelInputWithOriginalType(int32_t input_idx, const char_t *ori_type)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| input_idx | 输入 | 需要删除的输入边编号。 |
| ori_type | 输入 | 删除节点的原始算子类型。 |

#### 约束说明

无。
