# CastTrait<a name="ZH-CN_TOPIC_0000002554343749"></a>

类型转换模式结构体。

包括RegLayout、SatMode、MaskMergeMode、RoundMode。

```
struct CastTrait {
    RegLayout layoutMode = RegLayout::UNKNOWN;
    SatMode satMode = SatMode::UNKNOWN;
    MaskMergeMode mrgMode = MaskMergeMode::UNKNOWN;
    RoundMode roundMode = RoundMode::UNKNOWN;
}
```

