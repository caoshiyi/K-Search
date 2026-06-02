# SetNodeType

**页面ID:** atlasopapi_07_00040  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00040.html

---

#### 函数功能

设置算子的类型。

#### 函数原型

```
void SetNodeType(const ge::char_t *node_type)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| node_type | 输入 | 算子的类型。 |

#### 返回值说明

无。

#### 约束说明

无。

#### 调用示例

```
compute_node_info.SetNodeType("Const");
```
