# CalcTschBlockDim

**页面ID:** atlasascendc_api_07_1033  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_1033.html

---

#### 功能说明

针对Cube、Vector分离模式，用于计算Cube、Vector融合算子的blockDim。针对Vector/Cube融合计算的算子，启动时，按照AIV和AIC组合启动，blockDim用于设置启动多少个组合执行，比如某款AI处理器上有40个Vector核+20个Cube核，一个组合是2个Vector和1个Cube核，建议设置为20，此时会启动20个组合，即40个Vector和20个Cube核。使用该接口可以自动获取合适的blockDim值。

获取该值后，使用SetBlockDim进行blockDim的设置。

#### 函数原型

```
uint32_t CalcTschBlockDim(uint32_t sliceNum, uint32_t aicCoreNum, uint32_t aivCoreNum) const
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| sliceNum | 输入 | 数据切分的份数。 |
| aicCoreNum | 输入 | 如果算子实现使用了矩阵计算API，请传入GetCoreNumAic返回的数量，否则传入0。 |
| aivCoreNum | 输入 | 如果算子实现使用了矢量计算API，请传入GetCoreNumAiv返回的数量，否则传入0。 |

#### 返回值说明

返回用于底层任务调度的核数。

#### 约束说明

无

#### 调用示例

```
ge::graphStatus TilingXXX(gert::TilingContext* context) {
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto aicNum = ascendcPlatform.GetCoreNumAic();
    auto aivNum = ascendcPlatform.GetCoreNumAiv();
    // ..按照aivNum切分数据，并进行计算
    uint32_t sliceNum = aivNum;
    context->SetBlockDim(ascendcPlatform.CalcTschBlockDim(sliceNum, aicNum, aivNum));
    return ret;
}
```
