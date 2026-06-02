# GetInputTdInfo

**页面ID:** atlasopapi_07_00032  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00032.html

---

#### 函数功能

根据输入索引信息，获取算子的对应输入Tensor描述，注意，编译时无法确定的shape信息不在Tensor描述中（由于编译时无法确定shape，因此该Tensor描述里不包含shape信息）。

#### 函数原型

```
const CompileTimeTensorDesc *GetInputTdInfo(const size_t index) const
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| index | 输入 | 算子的输入索引，从0开始计数。 |

#### 返回值说明

返回const类型的Tensor描述信息。

#### 约束说明

无。

#### 调用示例

```
auto compute_node_info = extend_kernel_context->GetComputeNodeInfo();
auto input_td = compute_node_info->GetInputTdInfo(input_index);
```
