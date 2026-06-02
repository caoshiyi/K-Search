# GetCoreNumVector<a name="ZH-CN_TOPIC_0000002554343467"></a>

## 功能说明<a name="zh-cn_topic_0000001817714666_zh-cn_topic_0000001442758437_section36583473819"></a>

用于获取硬件平台独立的Vector Core的核数。

## 函数原型<a name="zh-cn_topic_0000001817714666_zh-cn_topic_0000001442758437_section13230182415108"></a>

```
uint32_t GetCoreNumVector(void) const
```

## 参数说明<a name="zh-cn_topic_0000001817714666_zh-cn_topic_0000001442758437_section189014013619"></a>

无

## 返回值说明<a name="zh-cn_topic_0000001817714666_zh-cn_topic_0000001442758437_section25791320141317"></a>

返回硬件平台Vector Core的核数。

## 约束说明<a name="zh-cn_topic_0000001817714666_zh-cn_topic_0000001442758437_section19165124931511"></a>

Ascend 950PR/Ascend 950DT不支持该接口，返回0

## 调用示例<a name="zh-cn_topic_0000001817714666_zh-cn_topic_0000001442758437_section320753512363"></a>

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

