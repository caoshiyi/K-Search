# TilingInputsDataDependency

**页面ID:** atlasopapi_07_00125  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00125.html

---

#### 函数功能

标记Tiling计算时需要依赖算子第几个输入tensor的值，同时标记tiling计算支持执行的位置。

#### 函数原型

```
OpImplRegisterV2 &TilingInputsDataDependency(std::initializer_list<int32_t> inputs)
OpImplRegisterV2 &TilingInputsDataDependency(std::initializer_list<int32_t> inputs, std::initializer_list<TilingPlacement> placements)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| inputs | 输入 | 指定算子tiling计算需要依赖的输入index列表。举例来说，inputs={0，3}，说明该算子的tiling计算需要依赖第0、3个输入的tensor值。 |
| placements | 输入 | 指定算子tiling计算可以执行的位置，0代表支持在host侧执行， 1代表支持在AI CPU上执行。如果不包含本参数，代表只支持在host执行。 |

#### 返回值说明

返回算子的OpImplRegisterV2对象，该对象新增注册了算子tiling值依赖输入的第index个tensor值以及可执行的位置。

#### 约束说明

无。
