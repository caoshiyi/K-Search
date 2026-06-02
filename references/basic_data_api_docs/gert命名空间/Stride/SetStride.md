# SetStride

**页面ID:** atlasopapi_07_00730  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00730.html

---

#### 函数功能

设置指定idx轴的步长值。

#### 函数原型

```
void SetStride(size_t idx, const int64_t stride)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| idx | 输入 | 维度的索引，调用者需要保证索引合法。 |
| stride | 输入 | 对idx轴设置的步长。 |

#### 返回值说明

无。

#### 约束说明

调用者需要保证index合法。

#### 调用示例

```
Stride stride0({3, 256, 256});
stride0.SetStride(0U, 1); // 1,256,256
```
