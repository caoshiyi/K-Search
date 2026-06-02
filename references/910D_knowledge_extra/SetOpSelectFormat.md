# SetOpSelectFormat<a name="ZH-CN_TOPIC_0000002554424251"></a>

## 功能说明<a name="zh-cn_topic_0000001626409757_zh-cn_topic_0000001526442954_zh-cn_topic_0000001525424352_section36583473819"></a>

如果您需要自行推导算子输入输出所支持的数据类型与格式，则可实现推导回调函数，并通过该接口进行注册。同时需要将[DynamicFormatFlag](DynamicFormatFlag.md)配置为true，则算子融合时会自动调用推导函数进行数据类型与格式的设置，算子原型注册时无需配置输入输出支持的数据类型与格式。

注意，如果算子原型已经注册过数据类型与格式，则以算子原型注册的数据类型与格式为准，即使注册了推导函数也不会执行。

## 函数原型<a name="zh-cn_topic_0000001626409757_zh-cn_topic_0000001526442954_zh-cn_topic_0000001525424352_section13230182415108"></a>

```
OpAICoreDef &SetOpSelectFormat(optiling::OP_CHECK_FUNC func)
```

## 参数说明<a name="zh-cn_topic_0000001626409757_zh-cn_topic_0000001526442954_zh-cn_topic_0000001525424352_section75395119104"></a>

<a name="zh-cn_topic_0000001626409757_zh-cn_topic_0000001526442954_zh-cn_topic_0000001575944081_table111938719446"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001626409757_zh-cn_topic_0000001526442954_zh-cn_topic_0000001575944081_row6223476444"><th class="cellrowborder" valign="top" width="17.22%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001626409757_zh-cn_topic_0000001526442954_zh-cn_topic_0000001575944081_p10223674448"><a name="zh-cn_topic_0000001626409757_zh-cn_topic_0000001526442954_zh-cn_topic_0000001575944081_p10223674448"></a><a name="zh-cn_topic_0000001626409757_zh-cn_topic_0000001526442954_zh-cn_topic_0000001575944081_p10223674448"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="15.340000000000002%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001626409757_zh-cn_topic_0000001526442954_zh-cn_topic_0000001575944081_p645511218169"><a name="zh-cn_topic_0000001626409757_zh-cn_topic_0000001526442954_zh-cn_topic_0000001575944081_p645511218169"></a><a name="zh-cn_topic_0000001626409757_zh-cn_topic_0000001526442954_zh-cn_topic_0000001575944081_p645511218169"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="67.44%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001626409757_zh-cn_topic_0000001526442954_zh-cn_topic_0000001575944081_p1922337124411"><a name="zh-cn_topic_0000001626409757_zh-cn_topic_0000001526442954_zh-cn_topic_0000001575944081_p1922337124411"></a><a name="zh-cn_topic_0000001626409757_zh-cn_topic_0000001526442954_zh-cn_topic_0000001575944081_p1922337124411"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001626409757_zh-cn_topic_0000001526442954_zh-cn_topic_0000001575944081_row152234713443"><td class="cellrowborder" valign="top" width="17.22%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001626409757_zh-cn_topic_0000001526442954_zh-cn_topic_0000001575944081_p318615392613"><a name="zh-cn_topic_0000001626409757_zh-cn_topic_0000001526442954_zh-cn_topic_0000001575944081_p318615392613"></a><a name="zh-cn_topic_0000001626409757_zh-cn_topic_0000001526442954_zh-cn_topic_0000001575944081_p318615392613"></a>func</p>
</td>
<td class="cellrowborder" valign="top" width="15.340000000000002%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001626409757_zh-cn_topic_0000001526442954_zh-cn_topic_0000001575944081_p320343694214"><a name="zh-cn_topic_0000001626409757_zh-cn_topic_0000001526442954_zh-cn_topic_0000001575944081_p320343694214"></a><a name="zh-cn_topic_0000001626409757_zh-cn_topic_0000001526442954_zh-cn_topic_0000001575944081_p320343694214"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.44%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001626409757_zh-cn_topic_0000001526442954_p12935163055011"><a name="zh-cn_topic_0000001626409757_zh-cn_topic_0000001526442954_p12935163055011"></a><a name="zh-cn_topic_0000001626409757_zh-cn_topic_0000001526442954_p12935163055011"></a>推导算子输入输出所支持数据类型与格式的函数。OP_CHECK_FUNC类型定义如下：</p>
<a name="zh-cn_topic_0000001626409757_zh-cn_topic_0000001526442954_screen746910291708"></a><a name="zh-cn_topic_0000001626409757_zh-cn_topic_0000001526442954_screen746910291708"></a><pre class="screen" codetype="Cpp" id="zh-cn_topic_0000001626409757_zh-cn_topic_0000001526442954_screen746910291708">using OP_CHECK_FUNC = ge::graphStatus (*)(const ge::Operator &amp;op, ge::AscendString &amp;result);</pre>
<p id="zh-cn_topic_0000001626409757_p1830771344310"><a name="zh-cn_topic_0000001626409757_p1830771344310"></a><a name="zh-cn_topic_0000001626409757_p1830771344310"></a>该函数的入参是算子的描述，包括算子的输入、输出、属性等信息，出参为包含了当前算子输入输出支持的数据类型与格式列表的字符串，字符串的格式样例如下：</p>
<pre class="screen" id="zh-cn_topic_0000001626409757_screen11586142152320"><a name="zh-cn_topic_0000001626409757_screen11586142152320"></a><a name="zh-cn_topic_0000001626409757_screen11586142152320"></a>{
    "input0": {"name": "x","dtype": "float16,float32,int32","format": "ND,ND,ND"},
    "input1": {"name": "y","dtype": "float16,float32,int32","format": "ND,ND,ND"},
    "output0": {"name": "z","dtype": "float16,float32,int32","format": "ND,ND,ND"}
}</pre>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0000001626409757_zh-cn_topic_0000001526442954_zh-cn_topic_0000001525424352_section25791320141317"></a>

OpAICoreDef算子定义，OpAICoreDef请参考[OpAICoreDef](OpAICoreDef.md)。

## 约束说明<a name="zh-cn_topic_0000001626409757_zh-cn_topic_0000001526442954_zh-cn_topic_0000001525424352_section19165124931511"></a>

无

## 调用示例<a name="zh-cn_topic_0000001626409757_zh-cn_topic_0000001526442954_zh-cn_topic_0000001575944081_section320753512363"></a>

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

