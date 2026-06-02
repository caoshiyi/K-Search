# IgnoreContiguous<a name="ZH-CN_TOPIC_0000002554343863"></a>

## 功能说明<a name="zh-cn_topic_0000001691759320_zh-cn_topic_0000001576875005_zh-cn_topic_0000001525424352_section36583473819"></a>

某些算子支持非连续的tensor，在算子的实现中对非连续的tensor做了转换处理。配置该参数后，框架会忽略对非连续的校验。

## 函数原型<a name="zh-cn_topic_0000001691759320_zh-cn_topic_0000001576875005_zh-cn_topic_0000001525424352_section13230182415108"></a>

```
OpParamDef &IgnoreContiguous(void)
```

## 参数说明<a name="zh-cn_topic_0000001691759320_zh-cn_topic_0000001576875005_zh-cn_topic_0000001525424352_section75395119104"></a>

无

## 返回值说明<a name="zh-cn_topic_0000001691759320_zh-cn_topic_0000001576875005_zh-cn_topic_0000001525424352_section25791320141317"></a>

OpParamDef算子定义，OpParamDef请参考[OpParamDef](OpParamDef.md)。

## 约束说明<a name="zh-cn_topic_0000001691759320_zh-cn_topic_0000001576875005_zh-cn_topic_0000001525424352_section19165124931511"></a>

无

