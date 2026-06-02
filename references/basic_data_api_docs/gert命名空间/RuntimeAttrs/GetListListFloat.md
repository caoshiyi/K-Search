# GetListListFloat

**页面ID:** atlasopapi_07_00147  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00147.html

---

#### 函数功能

获取ContinuousVectorVector *类型的属性值，即二维数组且每个元素类型为float。

#### 函数原型

```
const ContinuousVectorVector *GetListListFloat(const size_t index) const
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| index | 输入 | 属性在IR原型定义中的索引。 |

#### 返回值说明

指向属性值的指针。

#### 约束说明

无。

#### 调用示例

```
// 假设某算子的IR原型定义中，第一个属性的类型是二维数组float类型
const RuntimeAttrs * runtime_attrs = kernel_context->GetAttrs();
const ContinuousVectorVector *attr0 = runtime_attrs->GetListListFloat(0);
```
