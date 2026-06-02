# GetAttrs

**页面ID:** atlasopapi_07_00034  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00034.html

---

#### 函数功能

获取算子的属性值。

#### 函数原型

```
const RuntimeAttrs *GetAttrs() const
```

#### 参数说明

无。

#### 返回值说明

所有IR原型定义的属性值，为const类型的对象，属性值按照IR原型定义的顺序依次保存。

#### 约束说明

无。

#### 调用示例

```
auto compute_node_info = extend_kernel_context->GetComputeNodeInfo();
auto attrs = compute_node_info->GetAttrs();
```
