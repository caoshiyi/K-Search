# Allocator构造函数和析构函数

**页面ID:** atlasopapi_07_00278  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00278.html

---

#### 函数功能

Allocator构造函数和析构函数。

#### 函数原型

```
Allocator() = default
virtual ~Allocator() = default
Allocator(const Allocator &) = delete
Allocator &operator=(const Allocator &) = delete
```

#### 参数说明

无。

#### 返回值

无。

#### 异常处理

无。

#### 约束说明

纯虚类需要用户派生。
