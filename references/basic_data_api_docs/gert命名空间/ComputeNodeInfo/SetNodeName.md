# SetNodeName

**页面ID:** atlasopapi_07_00041  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00041.html

---

#### 函数功能

设置该ComputeNodeInfo对应的算子的名称。

#### 函数原型

```
void SetNodeName(const ge::char_t *node_name)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| node_name | 输入 | 算子的名称。 |

#### 返回值说明

无。

#### 约束说明

无。

#### 调用示例

```
compute_node_info->SetNodeName("Conv2d");
```
