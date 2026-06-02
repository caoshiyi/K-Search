# ReserveLocalMemory

**页面ID:** atlasascendc_api_07_00061  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_00061.html

---

#### 功能说明

该函数用于在Unified Buffer中预留指定大小的内存空间。调用该接口后，使用GetCoreMemSize可以获取实际可用的剩余Unified Buffer空间大小。

#### 函数原型

```
void ReserveLocalMemory(ReservedSize size)
```

#### 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| 需要预留的空间大小。 ``` enum class ReservedSize {     RESERVED_SIZE_8K,  // 预留8 * 1024B空间     RESERVED_SIZE_16K, // 预留16 * 1024B空间     RESERVED_SIZE_32K, // 预留32 * 1024B空间 }; ``` |  |  |

#### 约束说明

多次调用该函数时，仅保留最后一次调用的结果。

#### 调用示例

```
ge::graphStatus TilingXXX(gert::TilingContext* context) {
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ub_size, l1_size;
    // 预留8KB的Unified Buffer内存空间
    ascendcPlatform.ReserveLocalMemory(platform_ascendc::ReservedSize::RESERVED_SIZE_8K);
    // 获取Unified Buffer和L1的实际可用内存大小
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1_size);
    // ...
    return ret;
}
```

完整样例可参考与数学库高阶API配合使用的样例。
