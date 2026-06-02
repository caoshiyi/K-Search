# GetNodeName

**页面ID:** atlasopapi_07_00025  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00025.html

---

#### 函数功能

获取算子的名称。

#### 函数原型

```
const ge::char_t *GetNodeName() const
```

#### 参数说明

无。

#### 返回值说明

返回算子的名称。

#### 约束说明

无。

#### 调用示例

```
auto node_name = compute_node_info.GetNodeName();
```
