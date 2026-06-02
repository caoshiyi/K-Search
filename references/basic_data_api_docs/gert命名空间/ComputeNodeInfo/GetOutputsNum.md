# GetOutputsNum

**页面ID:** atlasopapi_07_00028  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00028.html

---

#### 函数功能

获取算子在网络中的实际输出个数。

#### 函数原型

```
size_t GetOutputsNum() const
```

#### 参数说明

无。

#### 返回值说明

算子的实际输出个数。

#### 约束说明

无。

#### 调用示例

```
size_t index = compute_node_info->GetOutputsNum();
```
