# MutableAttrs

**页面ID:** atlasopapi_07_00039  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00039.html

---

#### 函数功能

获取算子的属性值，仅在算子IR原型定义和调用IMPL_OP宏注册的属性值会被返回，其他属性值被丢弃。

本方法与GetAttrs的区别在于可以返回非const的属性对象。

#### 函数原型

```
RuntimeAttrs *MutableAttrs() const
```

#### 参数说明

无。

#### 返回值说明

所有IR原型定义过的属性值以及通过IMPL_OP宏注册的属性值，属性值按照IR原型定义的顺序依次保存。返回对象为非const。

#### 约束说明

无。

#### 调用示例

```
auto compute_node_info = extend_kernel_context->GetComputeNodeInfo();
auto attrs = compute_node_info->MutableAttrs();
```
