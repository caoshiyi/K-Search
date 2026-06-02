# OutputShapeDependOnCompute<a name="ZH-CN_TOPIC_0000002554423829"></a>

## 功能说明<a name="zh-cn_topic_0000002009944386_zh-cn_topic_0000001526594958_zh-cn_topic_0000001525424352_section36583473819"></a>

标识算子输出的shape是否依赖于计算得到。某些算子，比如NonZero（统计tensor中非零值的个数），计算完成前无法得知算子输出的shape信息，算子计算完成后才能获取。该类算子在原型定义时，需要使用OutputShapeDependOnCompute接口进行标识，同时在算子核函数中将实际输出shape写入到出参中，便于框架侧基于该信息进行输出内存的管理。对应的kernel侧实现请参考[输出shape依赖计算的算子kernel实现](Kernel侧算子实现.md#section1961152716108)。

## 函数原型<a name="zh-cn_topic_0000002009944386_zh-cn_topic_0000001526594958_zh-cn_topic_0000001525424352_section13230182415108"></a>

```
OpParamDef &OutputShapeDependOnCompute()
```

## 参数说明<a name="zh-cn_topic_0000002009944386_zh-cn_topic_0000001526594958_zh-cn_topic_0000001525424352_section75395119104"></a>

无

## 返回值说明<a name="zh-cn_topic_0000002009944386_zh-cn_topic_0000001526594958_zh-cn_topic_0000001525424352_section25791320141317"></a>

OpParamDef算子定义，OpParamDef请参考[OpParamDef](OpParamDef.md)。

## 约束说明<a name="zh-cn_topic_0000002009944386_zh-cn_topic_0000001526594958_zh-cn_topic_0000001525424352_section19165124931511"></a>

-   只能用于标识算子输出。
-   基于旧版本CANN包（不支持OutputShapeDependOnCompute特性）生成的自定义算子工程，无法兼容OutputShapeDependOnCompute接口。在使用非当前版本CANN包生成的自定义算子工程时，需特别注意兼容性问题。您可以通过查看自定义算子工程下cmake/util/ascendc\_impl\_build.py中有无output\_shape\_depend\_on\_compute字段来确认当前工程是否支持该特性，如果未找到该字段，则需要重新生成自定义算子工程以启用OutputShapeDependOnCompute特性。

## 调用示例<a name="zh-cn_topic_0000002009944386_zh-cn_topic_0000001526594958_zh-cn_topic_0000001575944081_section320753512363"></a>

```
this->Input("x1")
    .ParamType(REQUIRED)
    .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
    .Format({ge::FORMAT_ND, ge::FORMAT_ND});
this->Input("x2")
    .ParamType(REQUIRED)
    .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
    .Format({ge::FORMAT_ND, ge::FORMAT_ND});
this->Output("y1")
    .ParamType(REQUIRED)
    .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
    .Format({ge::FORMAT_ND, ge::FORMAT_ND})
    .OutputShapeDependOnCompute();
```

