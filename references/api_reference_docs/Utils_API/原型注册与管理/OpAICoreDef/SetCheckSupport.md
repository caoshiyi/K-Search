# SetCheckSupport

**页面ID:** atlasascendc_api_07_0981  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0981.html

---

#### 功能说明

如果您需要在算子融合阶段进行算子参数校验，则可实现算子参数校验回调函数，并通过该接口进行注册。同时，需要将NeedCheckSupportFlag参数配置为true，则算子编译和融合阶段会调用注册的算子参数校验函数进行相关信息的校验。

若算子参数校验函数校验通过，则代表AI Core支持此算子参数，会选择AI Core上相应的算子执行；否则，会继续查询AI CPU算子库然后执行。

#### 函数原型

```
OpAICoreDef &SetCheckSupport(optiling::OP_CHECK_FUNC func)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| 参数校验函数。OP_CHECK_FUNC类型定义如下： ``` using OP_CHECK_FUNC = ge::graphStatus (*)(const ge::Operator &op, ge::AscendString &result); ```  该函数的入参是算子的描述，包括算子的输入、输出、属性等信息，出参为包含了校验返回码和原因的字符串，字符串的格式如下： ``` {"ret_code": "1","reason": "your reason"} ``` 若校验成功，则函数返回ge::GRAPH_SUCCESS；若校验失败，则返回ge::GRAPH_FAILED。 |  |  |

#### 返回值说明

OpAICoreDef请参考OpAICoreDef。

#### 约束说明

无

#### 调用示例

下文展示了自定义Add算子参数校验函数实现和注册的样例。

- 参数校验函数实现如下：对第一个输入参数的shape进行校验，仅支持输入x shape的第一个维度为8，否则不支持。

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

- 参数校验函数注册的样例如下：

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
