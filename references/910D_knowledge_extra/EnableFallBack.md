# EnableFallBack<a name="ZH-CN_TOPIC_0000002523304118"></a>

## 功能说明<a name="zh-cn_topic_0000001600267337_zh-cn_topic_0000001526112350_zh-cn_topic_0000001525424352_section36583473819"></a>

通过本接口启用fallback配置，启用后将自动生成一个fallback函数并注册给GE。fallback函数的核心逻辑是将GE的输入、输出及属性转换为aclnn单算子API所需的参数格式，随后调用aclnn接口。动态图场景下，GE可直接调用fallback函数（函数中调用了aclnn接口），从而简化调度流程。关于fallback下发算子的详细介绍请参考《图模式开发指南》中的“自定义算子入图开发 \> 基于fallback形式下发算子”章节。

## 函数原型<a name="zh-cn_topic_0000001600267337_zh-cn_topic_0000001526112350_zh-cn_topic_0000001525424352_section13230182415108"></a>

```
OpDef &EnableFallBack(void)
```

## 参数说明<a name="zh-cn_topic_0000001600267337_zh-cn_topic_0000001526112350_zh-cn_topic_0000001525424352_section75395119104"></a>

无

## 返回值说明<a name="zh-cn_topic_0000001600267337_zh-cn_topic_0000001526112350_zh-cn_topic_0000001525424352_section25791320141317"></a>

OpDef算子定义，OpDef请参考[OpDef](OpDef.md)。

## 约束说明<a name="zh-cn_topic_0000001600267337_zh-cn_topic_0000001526112350_zh-cn_topic_0000001525424352_section19165124931511"></a>

-   算子需要注册并实现InferShape函数。
-   算子需要注册并实现InferDataType函数。

## 调用示例<a name="zh-cn_topic_0000001600267337_zh-cn_topic_0000001526112350_zh-cn_topic_0000001575944081_section320753512363"></a>

```
class AddCustom : public OpDef {
public:
    AddCustom(const char* name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("z").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND})
        this->AICore().AddConfig("ascendxxx");
        this->SetInferShape(ge::InferShapeFunc);
        this->SetInferDataType(ge::InferDataTypeFunc);
        this->EnableFallBack();
    }
};
OP_ADD(AddCustom);
```

