# GetSize

**页面ID:** atlasopapi_07_00049  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00049.html

---

#### 函数功能

获取当前存放的实际元素数量。

#### 函数原型

```
size_t GetSize() const
```

#### 参数说明

无。

#### 返回值说明

当前存放的实际元素数量。

#### 约束说明

无。

#### 调用示例

```
// 创建ContinuousVectorVector对象cvv
...
// 增加元素
...
auto cv = cvv->add(inner_vector_capacity);
...
// 获取当前存放的实际元素数量
auto size = cvv->GetSize();
```
