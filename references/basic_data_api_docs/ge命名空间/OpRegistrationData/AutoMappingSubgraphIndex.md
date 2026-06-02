# AutoMappingSubgraphIndex

**页面ID:** atlasopapi_07_00412  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00412.html

---

#### 函数功能

设置子图的输入输出和主图对应父节点输入输出的对应关系。

#### 函数原型

```
Status AutoMappingSubgraphIndex(const ge::Graph &graph,
const std::function<int32_t(int32_t data_index)> &input,
const std::function<int32_t(int32_t netoutput_index)> &output)
Status AutoMappingSubgraphIndex(const ge::Graph &graph,
const std::function<Status(int32_t data_index, int32_t &parent_input_index)> &input,
const std::function<Status(int32_t netoutput_index, int32_t &parent_output_index)> &output)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| graph | 输入 | 子图对象。 |
| input | 输入 | 输入对应关系函数。 |
| output | 输入 | 输出对应关系函数。 |

#### 约束说明

无。
