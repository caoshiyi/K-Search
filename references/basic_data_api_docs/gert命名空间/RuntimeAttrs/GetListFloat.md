# GetListFloat

**页面ID:** atlasopapi_07_00146  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00146.html

---

#### 函数功能

获取list_float32类型的属性值。

#### 函数原型

```
const TypedContinuousVector<float> *GetListFloat(const size_t index) const
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| index | 输入 | 属性在IR原型定义中以及在OP_IMPL注册中的索引。 |

#### 返回值说明

指向属性值的指针。

#### 约束说明

无。

#### 调用示例

```
const RuntimeAttrs * runtime_attrs = kernel_context->GetAttrs();
const TypedContinuousVector<float> *attr0 = runtime_attrs->GetListFloat(0);
```
