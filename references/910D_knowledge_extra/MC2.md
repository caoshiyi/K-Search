# MC2<a name="ZH-CN_TOPIC_0000002523304498"></a>

## 功能说明<a name="zh-cn_topic_0000001935420684_zh-cn_topic_0000001526594958_zh-cn_topic_0000001525424352_section36583473819"></a>

注册该算子是通算融合算子，调用该接口后会返回一个[OpMC2Def](OpMC2Def.md)结构，后续可通过该结构配置通信域名称。

## 函数原型<a name="zh-cn_topic_0000001935420684_zh-cn_topic_0000001526594958_zh-cn_topic_0000001525424352_section13230182415108"></a>

```
OpMC2Def &MC2(void)
```

## 参数说明<a name="zh-cn_topic_0000001935420684_zh-cn_topic_0000001526594958_zh-cn_topic_0000001525424352_section75395119104"></a>

无

## 返回值说明<a name="zh-cn_topic_0000001935420684_zh-cn_topic_0000001526594958_zh-cn_topic_0000001525424352_section25791320141317"></a>

[OpMC2Def](OpMC2Def.md)结构，后续可通过该结构配置通信域名称。

## 约束说明<a name="zh-cn_topic_0000001935420684_zh-cn_topic_0000001526594958_zh-cn_topic_0000001525424352_section19165124931511"></a>

基于旧版本CANN包（不支持MC2特性）生成的自定义算子工程，无法兼容MC2接口。在使用非当前版本CANN包生成的自定义算子工程时，需特别注意兼容性问题。您可以通过查看自定义算子工程下cmake/util/ascendc\_impl\_build.py中有无\_build\_mc2\_ctx字段来确认当前工程是否支持该特性，如果未找到该字段，则需要重新生成自定义算子工程以启用MC2特性。

