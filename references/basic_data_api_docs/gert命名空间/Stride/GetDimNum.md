# GetDimNum

**页面ID:** atlasopapi_07_00727  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00727.html

---

#### 函数功能

获取Stride对象的维度个数。

#### 函数原型

```
size_t GetDimNum() const
```

#### 参数说明

无

#### 返回值说明

Stride对象的维度个数。

#### 约束说明

无

#### 调用示例

```
Stride stride0({3, 256, 256});
auto dim_num = stride0.GetDimNum(); // 3
```
