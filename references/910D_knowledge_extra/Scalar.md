# Scalar<a name="ZH-CN_TOPIC_0000002554344143"></a>

## 功能说明<a name="zh-cn_topic_0000001759419021_zh-cn_topic_0000001526594958_zh-cn_topic_0000001525424352_section36583473819"></a>

配置该参数后，自动生成的单算子API（aclnnxxx）接口中，输入类型为aclScalar类型。

## 函数原型<a name="zh-cn_topic_0000001759419021_zh-cn_topic_0000001526594958_zh-cn_topic_0000001525424352_section13230182415108"></a>

```
OpParamDef &Scalar()
```

## 参数说明<a name="zh-cn_topic_0000001759419021_zh-cn_topic_0000001526594958_zh-cn_topic_0000001525424352_section75395119104"></a>

无

## 返回值说明<a name="zh-cn_topic_0000001759419021_zh-cn_topic_0000001526594958_zh-cn_topic_0000001525424352_section25791320141317"></a>

OpParamDef算子定义，OpParamDef请参考[OpParamDef](OpParamDef.md)。

## 约束说明<a name="zh-cn_topic_0000001759419021_zh-cn_topic_0000001526594958_zh-cn_topic_0000001525424352_section19165124931511"></a>

-   仅支持对算子输入做该参数配置，如果对算子输出配置该参数，则配置无效。
-   该接口仅在如下场景支持：
    -   通过单算子API执行的方式开发单算子调用应用。
    -   间接调用单算子API\(aclnnxxx\)接口：Pytorch框架单算子直调的场景。

## 调用示例<a name="zh-cn_topic_0000001759419021_zh-cn_topic_0000001526594958_zh-cn_topic_0000001575944081_section320753512363"></a>

```
this->Input("x")
    .Scalar()
```

