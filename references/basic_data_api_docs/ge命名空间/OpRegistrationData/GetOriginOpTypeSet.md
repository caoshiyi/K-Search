# GetOriginOpTypeSet

**页面ID:** atlasopapi_07_00400  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00400.html

---

#### 函数功能

获取原始模型的算子类型集合。

#### 函数原型

> **注意:** 

数据类型为string的接口后续版本会废弃，建议使用数据类型为非string的接口。

```
std::set<std::string> GetOriginOpTypeSet () const
Status GetOriginOpTypeSet(std::set<ge::AscendString> &ori_op_type) const
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| **ori_op_type** | 输出 | 原始模型的算子类型集合。 |

#### 约束说明

无。
