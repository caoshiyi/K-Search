# SetCheckSupport<a name="ZH-CN_TOPIC_0000002523343874"></a>

## 功能说明<a name="zh-cn_topic_0000001600467085_zh-cn_topic_0000001526442954_zh-cn_topic_0000001525424352_section36583473819"></a>

如果您需要在算子融合阶段进行算子参数校验，则可实现算子参数校验回调函数，并通过该接口进行注册。同时，需要将[NeedCheckSupportFlag](NeedCheckSupportFlag.md)参数配置为true，则算子编译和融合阶段会调用注册的算子参数校验函数进行相关信息的校验。

若算子参数校验函数校验通过，则代表AI Core支持此算子参数，会选择AI Core上相应的算子执行；否则，会继续查询AI CPU算子库然后执行。

## 函数原型<a name="zh-cn_topic_0000001600467085_zh-cn_topic_0000001526442954_zh-cn_topic_0000001525424352_section13230182415108"></a>

```
OpAICoreDef &SetCheckSupport(optiling::OP_CHECK_FUNC func)
```

## 参数说明<a name="zh-cn_topic_0000001600467085_zh-cn_topic_0000001526442954_zh-cn_topic_0000001525424352_section75395119104"></a>

<a name="zh-cn_topic_0000001600467085_zh-cn_topic_0000001526442954_zh-cn_topic_0000001575944081_table111938719446"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001600467085_zh-cn_topic_0000001526442954_zh-cn_topic_0000001575944081_row6223476444"><th class="cellrowborder" valign="top" width="17.22%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001600467085_zh-cn_topic_0000001526442954_zh-cn_topic_0000001575944081_p10223674448"><a name="zh-cn_topic_0000001600467085_zh-cn_topic_0000001526442954_zh-cn_topic_0000001575944081_p10223674448"></a><a name="zh-cn_topic_0000001600467085_zh-cn_topic_0000001526442954_zh-cn_topic_0000001575944081_p10223674448"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="15.340000000000002%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001600467085_zh-cn_topic_0000001526442954_zh-cn_topic_0000001575944081_p645511218169"><a name="zh-cn_topic_0000001600467085_zh-cn_topic_0000001526442954_zh-cn_topic_0000001575944081_p645511218169"></a><a name="zh-cn_topic_0000001600467085_zh-cn_topic_0000001526442954_zh-cn_topic_0000001575944081_p645511218169"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="67.44%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001600467085_zh-cn_topic_0000001526442954_zh-cn_topic_0000001575944081_p1922337124411"><a name="zh-cn_topic_0000001600467085_zh-cn_topic_0000001526442954_zh-cn_topic_0000001575944081_p1922337124411"></a><a name="zh-cn_topic_0000001600467085_zh-cn_topic_0000001526442954_zh-cn_topic_0000001575944081_p1922337124411"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001600467085_zh-cn_topic_0000001526442954_zh-cn_topic_0000001575944081_row152234713443"><td class="cellrowborder" valign="top" width="17.22%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001600467085_zh-cn_topic_0000001526442954_zh-cn_topic_0000001575944081_p318615392613"><a name="zh-cn_topic_0000001600467085_zh-cn_topic_0000001526442954_zh-cn_topic_0000001575944081_p318615392613"></a><a name="zh-cn_topic_0000001600467085_zh-cn_topic_0000001526442954_zh-cn_topic_0000001575944081_p318615392613"></a>func</p>
</td>
<td class="cellrowborder" valign="top" width="15.340000000000002%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001600467085_zh-cn_topic_0000001526442954_zh-cn_topic_0000001575944081_p320343694214"><a name="zh-cn_topic_0000001600467085_zh-cn_topic_0000001526442954_zh-cn_topic_0000001575944081_p320343694214"></a><a name="zh-cn_topic_0000001600467085_zh-cn_topic_0000001526442954_zh-cn_topic_0000001575944081_p320343694214"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.44%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001600467085_zh-cn_topic_0000001526442954_p12935163055011"><a name="zh-cn_topic_0000001600467085_zh-cn_topic_0000001526442954_p12935163055011"></a><a name="zh-cn_topic_0000001600467085_zh-cn_topic_0000001526442954_p12935163055011"></a>参数校验函数。OP_CHECK_FUNC类型定义如下：</p>
<a name="zh-cn_topic_0000001600467085_zh-cn_topic_0000001526442954_screen746910291708"></a><a name="zh-cn_topic_0000001600467085_zh-cn_topic_0000001526442954_screen746910291708"></a><pre class="screen" codetype="Cpp" id="zh-cn_topic_0000001600467085_zh-cn_topic_0000001526442954_screen746910291708">using OP_CHECK_FUNC = ge::graphStatus (*)(const ge::Operator &amp;op, ge::AscendString &amp;result);</pre>
<p id="zh-cn_topic_0000001600467085_p154394427555"><a name="zh-cn_topic_0000001600467085_p154394427555"></a><a name="zh-cn_topic_0000001600467085_p154394427555"></a>该函数的入参是算子的描述，包括算子的输入、输出、属性等信息，出参为包含了校验返回码和原因的字符串，字符串的格式如下：</p>
<pre class="screen" id="zh-cn_topic_0000001600467085_screen54391842185513"><a name="zh-cn_topic_0000001600467085_screen54391842185513"></a><a name="zh-cn_topic_0000001600467085_screen54391842185513"></a>{"ret_code": "1","reason": "your reason"}</pre>
<p id="zh-cn_topic_0000001600467085_p74391442175513"><a name="zh-cn_topic_0000001600467085_p74391442175513"></a><a name="zh-cn_topic_0000001600467085_p74391442175513"></a>若校验成功，则函数返回ge::GRAPH_SUCCESS；若校验失败，则返回ge::GRAPH_FAILED。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0000001600467085_zh-cn_topic_0000001526442954_zh-cn_topic_0000001525424352_section25791320141317"></a>

OpAICoreDef请参考[OpAICoreDef](OpAICoreDef.md)。

## 约束说明<a name="zh-cn_topic_0000001600467085_zh-cn_topic_0000001526442954_zh-cn_topic_0000001525424352_section19165124931511"></a>

无

## 调用示例<a name="zh-cn_topic_0000001600467085_zh-cn_topic_0000001526442954_zh-cn_topic_0000001575944081_section320753512363"></a>

下文展示了自定义Add算子参数校验函数实现和注册的样例。

-   参数校验函数实现如下：对第一个输入参数的shape进行校验，仅支持输入x shape的第一个维度为8，否则不支持。

    ```
    static ge::graphStatus CheckSupported(const ge::Operator &op, ge::AscendString &result)
    {
        std::string resultJsonStr;
        // 仅支持第一个输入参数shape的第一个维度为8，其他shape不支持
        if (op.GetInputDesc(0).GetShape().GetDim(0) == 8) {
            resultJsonStr = R"({"ret_code": "1","reason": "x.dim[0] is 8"})";
            result = ge::AscendString(resultJsonStr.c_str());
            return ge::GRAPH_SUCCESS;
        }
        resultJsonStr = R"({"ret_code": "0","reason": "xxx"})";
        result = ge::AscendString(resultJsonStr.c_str());
        return ge::GRAPH_FAILED;
    }
    ```

-   参数校验函数注册的样例如下：

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
                .SetOpSelectFormat(optiling::OpSelectFormat)
                .SetCheckSupport(optiling::CheckSupported);
    
            OpAICoreConfig aicConfig;
            aicConfig.DynamicCompileStaticFlag(true)
                .DynamicFormatFlag(true)
                .DynamicRankSupportFlag(true)
                .DynamicShapeSupportFlag(true)
                .NeedCheckSupportFlag(true)
                .PrecisionReduceFlag(true);
            // 注意：soc_version请替换成实际的AI处理器型号
            this->AICore().AddConfig("soc_version", aicConfig);
        }
    };
    ```

