# SetDimNum

**页面ID:** atlasopapi_07_00728  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00728.html

---

#### 函数功能

设置Stride对象的维度个数。

#### 函数原型

```
void SetDimNum(const size_t dim_num)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| dim_num | 输入 | 将Stride对象的维度个数设置为dim_num。 |

#### 返回值说明

无。

#### 约束说明

无。

#### 调用示例

```
Stride stride0({3, 256, 256});
stride0.SetDimNum(1);
auto dim_num = stride0.GetDimNum(); // 1
```
