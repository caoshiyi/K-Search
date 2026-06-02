# GetPlatformInfo

**页面ID:** atlasopapi_07_00246  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00246.html

---

#### 函数功能

获取PlatFormInfos指针。PlatformInfos类用于管理和查询硬件平台的相关信息，包括核数、内存带宽等。

#### 函数原型

**fe::PlatFormInfos *GetPlatformInfo() const**

#### 参数说明

无。

#### 返回值说明

PlatformInfos指针承载的是内部的硬件平台信息，这个信息作为PlatformAscendC的输入，开发者通过PlatformAscendC提供的方法获取硬件平台信息。

#### 约束说明

无。

#### 调用示例

```
ge::graphStatus TilingXXX(gert::TilingContext* context) {
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto coreNum = ascendcPlatform.GetCoreNum();
    // ... 根据核数自行设计Tiling策略
    context->SetBlockDim(coreNum);
    return ret;
}
```
