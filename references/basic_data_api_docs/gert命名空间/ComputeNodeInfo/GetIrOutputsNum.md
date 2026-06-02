# GetIrOutputsNum

**页面ID:** atlasopapi_07_00029  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00029.html

---

#### 函数功能

获取算子IR原型定义中的输出个数。

#### 函数原型

```
size_t GetIrOutputsNum() const
```

#### 参数说明

无。

#### 返回值说明

IR原型中定义的输出个数，size_t类型。

#### 约束说明

无。

#### 调用示例

```
size_t index = compute_node_info->GetIrOutputsNum();
```
