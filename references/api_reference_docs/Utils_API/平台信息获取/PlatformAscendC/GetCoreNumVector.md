# GetCoreNumVector

**页面ID:** atlasascendc_api_07_1032  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_1032.html

---

#### 功能说明

用于获取硬件平台独立的Vector Core的核数。

该接口仅在Atlas 推理系列产品有效，其他硬件平台型号均返回0。

#### 函数原型

```
uint32_t GetCoreNumVector(void) const
```

#### 参数说明

无

#### 返回值说明

返回硬件平台Vector Core的核数。

#### 约束说明

Atlas 训练系列产品，不支持该接口，返回0

Atlas 推理系列产品，支持该接口，返回硬件平台Vector Core的核数

Atlas A2 训练系列产品/Atlas A2 推理系列产品不支持该接口，返回0

Atlas A3 训练系列产品/Atlas A3 推理系列产品不支持该接口，返回0

Atlas 200I/500 A2 推理产品不支持该接口，返回0

#### 调用示例

```
ge::graphStatus TilingXXX(gert::TilingContext* context) {
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto aivCoreNum = ascendcPlatform.GetCoreNumAiv();
    auto vectorCoreNum = ascendcPlatform.GetCoreNumVector();
    auto allVecCoreNums = aivCoreNum + vectorCoreNum;
    // ...按照allVecCoreNums切分
    return ret;
}
```
