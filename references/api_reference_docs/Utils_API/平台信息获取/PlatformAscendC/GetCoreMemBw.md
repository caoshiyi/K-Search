# GetCoreMemBw

**页面ID:** atlasascendc_api_07_1035  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_1035.html

---

#### 功能说明

获取硬件平台存储空间的带宽大小。硬件存储空间类型定义如下：

```
enum class CoreMemType {
    L0_A = 0, // 预留参数，暂不支持
    L0_B = 1, // 预留参数，暂不支持
    L0_C = 2, // 预留参数，暂不支持
    L1 = 3,   // 预留参数，暂不支持
    L2 = 4,
    UB = 5,   // 预留参数，暂不支持
    HBM = 6,
    RESERVED
};
```

#### 函数原型

```
void GetCoreMemBw(const CoreMemType &memType, uint64_t &bwSize) const
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| memType | 输入 | 硬件存储空间类型。 |
| bwSize | 输出 | 对应硬件的存储空间的带宽大小。单位是Byte/cycle，cycle代表时钟周期。 |

#### 约束说明

无

#### 调用示例

```
ge::graphStatus TilingXXX(gert::TilingContext* context) {
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t l2_bw;
    ascendcPlatform.GetCoreMemBw(platform_ascendc::CoreMemType::L2, l2_bw);
    // ...
    return ret;
}
```
