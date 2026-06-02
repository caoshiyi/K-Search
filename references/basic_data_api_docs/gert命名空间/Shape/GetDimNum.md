# GetDimNum

**页面ID:** atlasopapi_07_00157  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00157.html

---

#### 函数功能

获取Shape的维度个数。

#### 函数原型

```
size_t GetDimNum() const
```

#### 参数说明

无。

#### 返回值说明

Shape的维度个数。

#### 约束说明

无。

#### 调用示例

```
Shape shape0({3, 256, 256});
auto dim_num = shape0.GetDimNum(); // 3
```
