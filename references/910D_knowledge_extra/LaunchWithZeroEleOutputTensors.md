# LaunchWithZeroEleOutputTensors<a name="ZH-CN_TOPIC_0000002523343588"></a>

## 功能说明<a name="zh-cn_topic_0000001575929572_zh-cn_topic_0000001526442954_zh-cn_topic_0000001525424352_section36583473819"></a>

在算子输出为全空Tensor时，用户可以配置该算子依旧会进行NPU上板执行。

## 函数原型<a name="zh-cn_topic_0000001575929572_zh-cn_topic_0000001526442954_zh-cn_topic_0000001525424352_section13230182415108"></a>

```
OpAICoreDef &OpAICoreDef::LaunchWithZeroEleOutputTensors(bool launchFlag)
```

## 参数说明<a name="zh-cn_topic_0000001575929572_zh-cn_topic_0000001526442954_zh-cn_topic_0000001525424352_section75395119104"></a>

<a name="zh-cn_topic_0000001575929572_zh-cn_topic_0000001526442954_zh-cn_topic_0000001575944081_table111938719446"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001575929572_zh-cn_topic_0000001526442954_zh-cn_topic_0000001575944081_row6223476444"><th class="cellrowborder" valign="top" width="17.22%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001575929572_zh-cn_topic_0000001526442954_zh-cn_topic_0000001575944081_p10223674448"><a name="zh-cn_topic_0000001575929572_zh-cn_topic_0000001526442954_zh-cn_topic_0000001575944081_p10223674448"></a><a name="zh-cn_topic_0000001575929572_zh-cn_topic_0000001526442954_zh-cn_topic_0000001575944081_p10223674448"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="15.340000000000002%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001575929572_zh-cn_topic_0000001526442954_zh-cn_topic_0000001575944081_p645511218169"><a name="zh-cn_topic_0000001575929572_zh-cn_topic_0000001526442954_zh-cn_topic_0000001575944081_p645511218169"></a><a name="zh-cn_topic_0000001575929572_zh-cn_topic_0000001526442954_zh-cn_topic_0000001575944081_p645511218169"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="67.44%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001575929572_zh-cn_topic_0000001526442954_zh-cn_topic_0000001575944081_p1922337124411"><a name="zh-cn_topic_0000001575929572_zh-cn_topic_0000001526442954_zh-cn_topic_0000001575944081_p1922337124411"></a><a name="zh-cn_topic_0000001575929572_zh-cn_topic_0000001526442954_zh-cn_topic_0000001575944081_p1922337124411"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001575929572_row261104521415"><td class="cellrowborder" valign="top" width="17.22%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001575929572_p13611145151412"><a name="zh-cn_topic_0000001575929572_p13611145151412"></a><a name="zh-cn_topic_0000001575929572_p13611145151412"></a>launchFlag</p>
</td>
<td class="cellrowborder" valign="top" width="15.340000000000002%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001575929572_p9611345141418"><a name="zh-cn_topic_0000001575929572_p9611345141418"></a><a name="zh-cn_topic_0000001575929572_p9611345141418"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.44%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001575929572_p14611945131415"><a name="zh-cn_topic_0000001575929572_p14611945131415"></a><a name="zh-cn_topic_0000001575929572_p14611945131415"></a>用户开发的自定义算子，在所有输出都为空Tensor时，如果需要该算子进行NPU上板执行时，需要配置为true，否则不会执行该算子。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0000001575929572_zh-cn_topic_0000001526442954_zh-cn_topic_0000001525424352_section25791320141317"></a>

OpAICoreDef算子定义，OpAICoreDef请参考[OpAICoreDef](OpAICoreDef.md)。

## 约束说明<a name="zh-cn_topic_0000001575929572_zh-cn_topic_0000001526442954_zh-cn_topic_0000001525424352_section19165124931511"></a>

无

## 调用示例<a name="zh-cn_topic_0000001575929572_zh-cn_topic_0000001526442954_zh-cn_topic_0000001575944081_section320753512363"></a>

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
            .SetCheckSupport(optiling::CheckSupported)
            .LaunchWithZeroEleOutputTensors(true);

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

