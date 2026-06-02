# operator[]

**页面ID:** atlasopapi_07_00154  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00154.html

---

#### 函数功能

获取指定index轴的dim值。

#### 函数原型

```
const int64_t &operator const
int64_t &operator
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| idx | 输入 | dim的index，调用者需要保证index合法。 |

#### 返回值说明

- const int64_t &operator const：dim值，在idx>=kMaxDimNum时，行为未定义。
- int64_t &operator：dim值，在idx>=kMaxDimNum时，行为未定义。

#### 约束说明

调用者需要保证index合法，即idx<kMaxDimNum。

#### 调用示例

```
Shape shape0({3, 256, 256});
auto dim0 = shape0[0]; // 3
auto dim5 = shape0[5]; // 0
auto invalid_dim = shape0[kMaxDimNum]; // 行为未定义
```
