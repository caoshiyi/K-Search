# operator==

**页面ID:** atlasopapi_07_00063  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00063.html

---

#### 函数功能

判断本补维规则对象与另一个对象是否一致。

#### 函数原型

```
bool operator==(const ExpandDimsType &other) const
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| other | 输入 | 另一个补维规则对象。 |

#### 返回值说明

true表示一致，false表示不一致。

#### 约束说明

无。

#### 调用示例

```
ExpandDimsType type1("1001");
ExpandDimsType type2("1001");
bool is_same_type = type1 == type2; // true
```
