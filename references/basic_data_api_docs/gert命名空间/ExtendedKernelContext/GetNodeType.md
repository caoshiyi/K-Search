# GetNodeType

**页面ID:** atlasopapi_07_00081  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00081.html

---

#### 函数功能

获取算子的类型。

#### 函数原型

```
const ge::char_t *GetNodeType() const
```

#### 参数说明

无。

#### 返回值说明

算子的类型。

#### 约束说明

无。

#### 调用示例

```
// 假设已存在KernelContext *context
auto extend_context = reinterpret_cast<ExtendedKernelContext *>(context);
auto node_type = extend_context->GetNodeType();
```
