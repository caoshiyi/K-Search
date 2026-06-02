# GetCoreMemSize

**页面ID:** atlasascendc_api_07_1034  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_1034.html

---

#### 功能说明

获取硬件平台存储空间的内存大小，例如L1、L0_A、L0_B、L2等，支持的存储空间类型定义如下：

```
enum class CoreMemType {
L0_A = 0, // L0A Buffer
L0_B = 1, // L0B Buffer
L0_C = 2, // L0C Buffer
L1 = 3,   // L1 Buffer
L2 = 4,   // L2 Cache
UB = 5,   // Unified Buffer
HBM = 6,  // GM
FB = 7,   // Fixpipe Buffer
BT = 8,   // BiasTable Buffer
RESERVED
};
```

#### 函数原型

```
void GetCoreMemSize(const CoreMemType &memType, uint64_t &size) const
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| memType | 输入 | 硬件存储空间类型。 |
| size | 输出 | 对应类型的存储空间大小，单位：字节。 |

#### 约束说明

无

#### 调用示例

```
ge::graphStatus TilingXXX(gert::TilingContext* context) {
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ub_size, l1_size;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1_size);
    // ...
    return ret;
}
```
