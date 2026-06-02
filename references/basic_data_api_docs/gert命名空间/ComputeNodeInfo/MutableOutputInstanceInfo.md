# MutableOutputInstanceInfo

**页面ID:** atlasopapi_07_00036  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00036.html

---

#### 函数功能

根据算子IR原型中的输出索引，获取对应的实例化对象。

#### 函数原型

```
AnchorInstanceInfo *MutableOutputInstanceInfo(const size_t ir_index) const
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| ir_index | 输入 | 算子IR原型定义中的输出索引，从0开始计数。 |

#### 返回值说明

返回的实例化对象的地址。返回对象为非const。

#### 约束说明

无。

#### 调用示例

```
for (size_t i = 0; i < ir_outputs.size(); ++i) {
  auto ins_info = compute_node_info.MutableOutputInstanceInfo(i);
  ...
}
```
