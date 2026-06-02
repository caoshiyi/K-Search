# SetInstantiationNum

**页面ID:** atlasopapi_07_00009  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00009.html

---

#### 函数功能

设置IR原型定义某个输入对应的实际输入个数。

#### 函数原型

```
void SetInstantiationNum(const uint32_t instantiation_num)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| instantiation_num | 输入 | 实例化的个数。 |

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
  size_t instance_num = ir_index_to_instance_index_pair_map[i].second; // 获取统计后的算子IR输入对应的实际输入个数
  ins_info->SetInstantiationNum(instance_num); // 将该信息保存到IR输入对应的AnchorInstanceInfo对象中
}
```
