# GetCoreNumAic

**页面ID:** atlasascendc_api_07_1030  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_1030.html

---

#### 功能说明

获取当前硬件平台AI Core中Cube核数。若AI Core的架构为Cube、Vector分离模式，返回Cube Core的核数；耦合模式返回AI Core的核数。

#### 函数原型

```
uint32_t GetCoreNumAic(void) const
```

#### 参数说明

无

#### 返回值说明

Atlas 训练系列产品，耦合模式，返回AI Core的核数

Atlas 推理系列产品，耦合模式，返回AI Core的核数

Atlas A2 训练系列产品/Atlas A2 推理系列产品，分离模式，返回Cube Core的核数

Atlas A3 训练系列产品/Atlas A3 推理系列产品，分离模式，返回Cube Core的核数

#### 约束说明

无

#### 调用示例

```
ge::graphStatus TilingXXX(gert::TilingContext* context) {
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto aicNum = ascendcPlatform.GetCoreNumAic();
    auto aivNum = ascendcPlatform.GetCoreNumAiv();
    // ...按照aivNum切分
    context->SetBlockDim(ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum));
    return ret;
}
```
