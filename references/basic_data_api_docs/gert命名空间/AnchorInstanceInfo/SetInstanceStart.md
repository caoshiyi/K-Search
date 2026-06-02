# SetInstanceStart

**页面ID:** atlasopapi_07_00008  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00008.html

---

#### 函数功能

设置算子某个IR输入在实际输入中的起始序号（index）。

#### 函数原型

```
void SetInstanceStart(const uint32_t instance_start)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| instance_start | 输入 | 首个实例化Anchor的index。 |

#### 返回值说明

无。

#### 约束说明

无。

#### 调用示例

```
const auto &ir_inputs = node->GetOpDesc()->GetIrInputs();  // 算子IR原型定义的所有输入
for (size_t i = 0; i < ir_inputs.size(); ++i) {
  auto ins_info = compute_node_info.MutableInputInstanceInfo(i);  // 获取第i个IR输入对应的AnchorInstanceInfo对象
  GE_ASSERT_NOTNULL(ins_info);
  size_t input_index = ir_index_to_instance_index_pair_map[i].first; // 获取统计后的算子IR输入对应的实际输入index
  ins_info->SetInstanceStart(input_index); // 将该信息保存到IR输入对应的AnchorInstanceInfo对象中
}
```
