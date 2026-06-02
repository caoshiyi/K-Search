# Get

**页面ID:** atlasopapi_07_00048  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00048.html

---

#### 函数功能

获取第index个元素的首地址。

#### 函数原型

```
const ContinuousVector *Get(const size_t index) const
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| index | 输入 | 元素index。 |

#### 返回值说明

第index个元素的首地址。

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
// 获取第0个元素的首地址
auto cv1 = cvv->Get(0U);
```
