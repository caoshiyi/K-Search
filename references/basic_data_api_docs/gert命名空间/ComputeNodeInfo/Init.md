# Init

**页面ID:** atlasopapi_07_00042  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00042.html

---

#### 函数功能

初始化ComputeNodeInfo类。

#### 函数原型

```
void Init(const size_t ir_inputs_num, const size_t inputs_num, const size_t outputs_num, const ge::char_t *node_name, const ge::char_t *node_type)
void Init(const size_t ir_inputs_num, const size_t ir_outputs_num, const size_t inputs_num, const size_t outputs_num, const size_t attr_size, const ge::char_t *node_name, const ge::char_t *node_type)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| **ir_inputs_num** | 输入 | 算子原型输入的个数。 |
| **inputs_num** | 输入 | 算子实际输入个数。 |
| **outputs_num** | 输入 | 算子实际输出个数。 |
| **node_name** | 输入 | 节点名称。 |
| **node_type** | 输入 | 节点类型（算子原型名称）。 |
| **ir_outputs_num** | 输入 | 算子原型输出的个数。 |
| **attr_size** | 输入 | 属性个数。 |

#### 返回值说明

无。

#### 约束说明

无。

#### 调用示例

```
auto ir_input_num = compute_node->GetOpDesc()->GetIrInputs().size();
auto inputs_num = compute_node->GetInDataNodesAndAnchors().size();
auto outputs_num = compute_node->GetOutDataNodesAndAnchors().size();
auto compute_node_info_holder = std::make_unique<uint8_t[]>(total_size);
auto compute_node_info = reinterpret_cast<ComputeNodeInfo *>(compute_node_info_holder.get());
compute_node_info->Init(ir_input_num, inputs_num, outputs_num, node_name, node_type);
```
