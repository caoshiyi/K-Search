# SetOpSelectFormat

**页面ID:** atlasascendc_api_07_0982  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0982.html

---

#### 功能说明

如果您需要自行推导算子输入输出所支持的数据类型与格式，则可实现推导回调函数，并通过该接口进行注册。同时需要将DynamicFormatFlag配置为true，则算子融合时会自动调用推导函数进行数据类型与格式的设置，算子原型注册时无需配置输入输出支持的数据类型与格式。

注意，如果算子原型已经注册过数据类型与格式，则以算子原型注册的数据类型与格式为准，即使注册了推导函数也不会执行。

#### 函数原型

```
OpAICoreDef &SetOpSelectFormat(optiling::OP_CHECK_FUNC func)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| 推导算子输入输出所支持数据类型与格式的函数。OP_CHECK_FUNC类型定义如下： ``` using OP_CHECK_FUNC = ge::graphStatus (*)(const ge::Operator &op, ge::AscendString &result); ```  该函数的入参是算子的描述，包括算子的输入、输出、属性等信息，出参为包含了当前算子输入输出支持的数据类型与格式列表的字符串，字符串的格式样例如下： ``` {     "input0": {"name": "x","dtype": "float16,float32,int32","format": "ND,ND,ND"},     "input1": {"name": "y","dtype": "float16,float32,int32","format": "ND,ND,ND"},     "output0": {"name": "z","dtype": "float16,float32,int32","format": "ND,ND,ND"} } ``` |  |  |

#### 返回值说明

OpAICoreDef算子定义，OpAICoreDef请参考OpAICoreDef。

#### 约束说明

无

#### 调用示例

如下展示了自定义Add算子推导函数实现和注册的样例。

```
static ge::graphStatus OpSelectFormat(const ge::Operator &op, ge::AscendString &result)
{
    std::string resultJsonStr;
    // 如果本次执行第一个输入参数shape的第一个维度<=8，则支持更多的格式，否则仅支持int32
    if (op.GetInputDesc(0).GetShape().GetDim(0) <= 8) {
        resultJsonStr = R"({
        "input0": {"name": "x","dtype": "float16,float32,int32","format": "ND,ND,ND","unknownshape_format": "ND,ND,ND"},
        "input1": {"name": "y","dtype": "float16,float32,int32","format": "ND,ND,ND","unknownshape_format": "ND,ND,ND"},
        "output0": {"name": "z","dtype": "float16,float32,int32","format": "ND,ND,ND","unknownshape_format": "ND,ND,ND"}
        })";
    } else {
        resultJsonStr = R"({
        "input0": {"name": "x","dtype": "int32","format": "ND","unknownshape_format": "ND"},
        "input1": {"name": "y","dtype": "int32","format": "ND","unknownshape_format": "ND"},
        "output0": {"name": "z","dtype": "int32","format": "ND","unknownshape_format": "ND"}
        })";
    }
    result = ge::AscendString(resultJsonStr.c_str());
    return ge::GRAPH_SUCCESS;
}
```

推导函数的注册样例如下：

```
class AddCustom : public OpDef {
public:
    AddCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED);
        this->Input("y")
            .ParamType(REQUIRED);
        this->Output("z")
            .ParamType(REQUIRED);
        this->SetInferShape(ge::InferShape);
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .SetTilingParse(optiling::TilingPrepare)
            .SetOpSelectFormat(optiling::OpSelectFormat);

        OpAICoreConfig aicConfig;
        aicConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true);
        // 注意：soc_version请替换成实际的AI处理器型号
        this->AICore().AddConfig("soc_version", aicConfig);
    }
};
```
