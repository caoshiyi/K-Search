# operator[]

**页面ID:** atlasopapi_07_00726  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00726.html

---

#### 函数功能

获取指定idx轴的步长值。

#### 函数原型

```
const int64_t &operator const
int64_t &operator
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| idx | 输入 | 维度的索引，调用者需要保证索引合法。 |

#### 返回值说明

- const int64_t &operator const：步长值，在idx>=kMaxDimNum时，行为未定义。
- int64_t &operator：dim值，在idx>=kMaxDimNum时，行为未定义。

#### 约束说明

调用者需要保证index合法，即idx<kMaxDimNum。

#### 调用示例

```
Stride stride({3, 256, 256});
auto str0 = stride[0]; // 3
auto str5 = stride[5]; // 0
auto invalid_str = stride[Stride::kMaxDimNum]; // 行为未定义
```
