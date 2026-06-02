# SetCommEngine

**页面ID:** atlasascendc_api_07_10399  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_10399.html

---

#### 功能说明

设置通信任务使用的通信引擎。

#### 函数原型

```
uint32_t SetCommEngine(uint8_t commEngine)
```

#### 参数说明

**表1 **参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| commEngine | 输入 | 通信引擎。uint8_t类型，该参数的取值范围请参考：表2。 |

#### 返回值说明

- 0表示设置成功。
- 非0表示设置失败。

#### 约束说明

无

#### 调用示例

```
static ge::graphStatus AllToAllVCustomTilingFunc(gert::TilingContext *context)
{
    AllToAllVCustomV3TilingData *tiling = context->GetTilingData<AllToAllVCustomV3TilingData>();
    const std::string groupName = "testGroup";
    const std::string algConfig = "AlltoAll=level0:fullmesh;level1:pairwise";
    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(groupName, HCCL_CMD_ALLTOALLV, algConfig, 0);
    mc2CcTilingConfig.SetCommEngine(1U);
    mc2CcTilingConfig.GetTiling(tiling->mc2InitTiling);
    mc2CcTilingConfig.GetTiling(tiling->mc2CcTiling);
    return ge::GRAPH_SUCCESS;
}
```
