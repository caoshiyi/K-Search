# operator!=

**页面ID:** atlasopapi_07_00725  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00725.html

---

#### 函数功能

判断与另一个Stride对象是否不等。

#### 函数原型

```
bool operator!=(const Stride &rht) const
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| rht | 输入 | 另一个Stride对象。 |

#### 返回值说明

true：不相等；false：相等。

#### 约束说明

无

#### 调用示例

```
Stride stride0({3, 256, 256});
Stride stride1({1, 3, 256, 256});
auto is_diff_shape = stride0 != stride1; // 返回值为true，不相等
```
