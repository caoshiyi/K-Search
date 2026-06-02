# GetSocVersion<a name="ZH-CN_TOPIC_0000002523343982"></a>

## 功能说明<a name="zh-cn_topic_0000001664705472_zh-cn_topic_0000001442758437_section36583473819"></a>

获取当前硬件平台版本型号。

## 函数原型<a name="zh-cn_topic_0000001664705472_zh-cn_topic_0000001442758437_section13230182415108"></a>

```
SocVersion GetSocVersion(void) const
```

## 参数说明<a name="zh-cn_topic_0000001664705472_zh-cn_topic_0000001442758437_section189014013619"></a>

无

## 返回值<a name="zh-cn_topic_0000001664705472_zh-cn_topic_0000001442758437_section25791320141317"></a>

当前硬件平台版本型号的枚举类。该枚举类和AI处理器型号的对应关系请通过CANN软件安装后文件存储路径下include/tiling/platform/platform\_ascendc.h头文件获取。

## 约束说明<a name="zh-cn_topic_0000001664705472_zh-cn_topic_0000001442758437_section19165124931511"></a>

无

## 调用示例<a name="zh-cn_topic_0000001664705472_zh-cn_topic_0000001442758437_section320753512363"></a>

```
ge::graphStatus TilingXXX(gert::TilingContext* context) {
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto socVersion = ascendcPlatform.GetSocVersion();
    // 根据所获得的版本型号自行设计Tiling策略
    // ASCENDXXX请替换为实际的版本型号
    if (socVersion == platform_ascendc::SocVersion::ASCENDXXX) {
        // ...
    }
    return ret;
}
```

