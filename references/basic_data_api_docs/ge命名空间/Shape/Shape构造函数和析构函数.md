# Shape构造函数和析构函数

**页面ID:** atlasopapi_07_00422  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00422.html

---

#### 函数功能

Shape构造函数和析构函数。

#### 函数原型

```
Shape()
~Shape() = default
explicit Shape(const std::vector<int64_t> &dims)
```

#### 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| dims | 输入 | Shape的取值内容。 Shape表征张量数据的维度大小，用std::vector<int64_t>表征每一个维度的具体大小。 |

#### 返回值

Shape构造函数返回Shape类型的对象。

#### 异常处理

无。

#### 约束说明

无。
