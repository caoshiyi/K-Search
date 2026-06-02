# FrameworkRegistry类构造函数和析构函数

**页面ID:** atlasopapi_07_00295  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00295.html

---

#### 函数功能

FrameworkRegistry构造函数和析构函数。

#### 函数原型

```
FrameworkRegistry(const FrameworkRegistry &) = delete
FrameworkRegistry& operator = (const FrameworkRegistry &) = delete
~FrameworkRegistry()
static FrameworkRegistry& Instance()
```

#### 参数说明

NA

#### 返回值

Instance()，返回FrameworkRegistry的单例对象。

#### 异常处理

无。

#### 约束说明

无。
