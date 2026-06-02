# InputsDataDependency

**页面ID:** atlasopapi_07_00121  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00121.html

---

#### 函数功能

设置算子计算依赖第几个输入tensor的值。

所谓的数据依赖，是指算子的计算不仅依赖于输入tensor的shape，还依赖输入tensor的具体值。

#### 函数原型

```
OpImplRegisterV2 &InputsDataDependency(std::initializer_list<int32_t> inputs)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| inputs | 输入 | 指定算子计算需要依赖的输入index列表。举例来说，inputs={0，3}，说明该算子的计算需要依赖第0、3个输入的tensor值。 |

#### 返回值说明

返回算子的OpImplRegisterV2 对象，该对象新增设置算子计算依赖第几个输入tensor的值。

#### 约束说明

无。
