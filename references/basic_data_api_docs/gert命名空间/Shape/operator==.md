# operator==

**页面ID:** atlasopapi_07_00152  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00152.html

---

#### 函数功能

判断与另外一个shape对象是否相等，如果两个shape的dim num相等，并且dim num内每个dim的值都相等，则认为两个shape相等。

#### 函数原型

```
bool operator==(const Shape &rht) const
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| rht | 输入 | 另一个Shape对象。 |

#### 返回值说明

true：相等；false：不相等。

#### 约束说明

无。

#### 调用示例

```
Shape shape0({3, 256, 256});
Shape shape1({1, 3, 256, 256});
auto is_same_shape = shape0 == shape1; // 返回值为false，不相等
```
