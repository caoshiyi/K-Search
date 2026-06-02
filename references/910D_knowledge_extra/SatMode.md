# SatMode<a name="ZH-CN_TOPIC_0000002523303688"></a>

设置Cast类型转换饱和与不饱和模式，取值如下：

-   SatMode::UNKNOWN，当转换类型不支持SatMode模式时，选择该模式；
-   SatMode::NO\_SAT，不饱和模式；
-   SatMode::SAT，饱和模式。

```
enum class SatMode {
    UNKNOWN = -1,
    NO_SAT,
    SAT
};
```

