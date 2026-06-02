# GetOutputTdInfo

**页面ID:** atlasopapi_07_00033  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00033.html

---

#### 函数功能

获取算子指定输出的Tensor描述，注意，编译时无法确定的shape信息不在Tensor描述中（由于编译时无法确定shape，因此该Tensor描述里不包含shape信息）。

#### 函数原型

```
const CompileTimeTensorDesc *GetOutputTdInfo(const size_t index) const
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| index | 输入 | 算子的输出索引，从0开始计数。 |

#### 返回值说明

返回const类型的Tensor描述信息。

#### 约束说明

无。

#### 调用示例

```
auto compute_node_info = extend_kernel_context->GetComputeNodeInfo();
auto output_td = compute_node_info->GetOutputTdInfo(output_index);
```
