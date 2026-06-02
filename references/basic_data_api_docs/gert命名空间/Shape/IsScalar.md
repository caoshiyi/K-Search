# IsScalar

**页面ID:** atlasopapi_07_00155  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00155.html

---

#### 函数功能

判断本shape是否为标量，所谓标量，是指GetDimNum()为0。

#### 函数原型

```
bool IsScalar() const
```

#### 参数说明

无。

#### 返回值说明

true/false。

#### 约束说明

无。

#### 调用示例

```
Shape shape0({3, 256, 256});
Shape shape2;
shape0.IsScalar(); // false
shape2.IsScalar(); // true
```
