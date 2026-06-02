# SetDim

**页面ID:** atlasopapi_07_00160  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00160.html

---

#### 函数功能

设置某一个轴的维度值。

#### 函数原型

```
void SetDim(size_t idx, const int64_t dim_value)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| idx | 输入 | dim的index，调用者需要保证index合法。 |
| dim_value | 输入 | 对idx轴设置的维度值。 |

#### 返回值说明

无。

#### 约束说明

调用者需要保证index合法。

#### 调用示例

```
Shape shape0({3, 256, 256});
shape0.SetDim(0U, 1); // 1,256,256
```
