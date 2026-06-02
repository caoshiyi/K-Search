# SetShapeRange

**页面ID:** atlasopapi_07_00454  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00454.html

---

#### 函数功能

设置shape的变化范围。

#### 函数原型

```
graphStatus SetShapeRange(const std::vector<std::pair<int64_t,int64_t>> &range)
```

#### 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| range | 输入 | shape代表的变化范围。vector中的每一个元素为一个pair，pair的第一个值为该维度上的dim最小值，第二个值为该维度上dim的最大值。举例如下： 该tensor的shape为{1,1,-1,2}，第三个轴的最大值为100，则range可设置为{{1,1},{1,1},{1,100},{2,2}}。 |

#### 返回值

graphStatus类型：若成功，则该值为GRAPH_SUCCESS(即0)，其他值则为执行失败。

#### 异常处理

无。

#### 约束说明

无。
