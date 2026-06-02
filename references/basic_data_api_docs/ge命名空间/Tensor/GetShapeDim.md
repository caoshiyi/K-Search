# GetShapeDim

**页面ID:** atlasopapi_07_00472  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00472.html

---

#### 函数功能

获取Shape第idx维度。

#### 函数原型

```
int64_t GetShapeDim(const size_t idx) const
```

#### 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| idx | 输入 | 维度的索引，索引从0开始。 |

#### 返回值

返回Shape第idx位置的值，默认值为0。

#### 异常处理

无。

#### 约束说明

无。
