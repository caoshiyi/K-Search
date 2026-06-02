# GetStride

**页面ID:** atlasopapi_07_00729  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00729.html

---

#### 函数功能

获取对应idx轴的步长值。

#### 函数原型

```
int64_t GetStride(const size_t idx) const
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| idx | 输入 | 维度的索引，调用者需要保证索引合法。 |

#### 返回值说明

步长值，在idx>=kMaxDimNum时，返回`kInvalidStrideValue`。

#### 约束说明

调用者需要保证idx合法，即idx<kMaxDimNum。

#### 调用示例

```
Stride stride({3, 256, 256});
auto stride0 = stride.GetStride(0); // 3
auto invalid_stride = stride.GetStride(Stride::kMaxDimNum); // kInvalidStrideValue
```
