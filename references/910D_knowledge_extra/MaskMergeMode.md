# MaskMergeMode<a name="ZH-CN_TOPIC_0000002523343742"></a>

决定mask未选择元素对于dst元素是否置0，取值如下：

-   MaskMergeMode::UNKNOWN，当转换类型不支持MaskMergeMode模式时，选择该模式；
-   MaskMergeMode::ZEROING，mask未选择的元素在dst中置零；
-   MaskMergeMode::MERGING，mask未选择的元素对应dst元素中保留dst原值。

```
enum class MaskMergeMode {
    UNKNOWN,
    MERGING,
    ZEROING
};
```

