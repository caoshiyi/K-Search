# LaunchWithZeroEleOutputTensors

**页面ID:** atlasascendc_api_07_00187  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_00187.html

---

#### 功能说明

在算子输出为全空Tensor时，用户可以配置该算子依旧会进行NPU上板执行。

#### 函数原型

```
OpAICoreDef &OpAICoreDef::LaunchWithZeroEleOutputTensors(bool launchFlag)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| launchFlag | 输入 | 用户开发的自定义算子，在所有输出都为空Tensor时，如果需要该算子进行NPU上板执行时，需要配置为true，否则不会执行该算子。 |

#### 返回值说明

OpAICoreDef算子定义，OpAICoreDef请参考OpAICoreDef。

#### 约束说明

无

#### 调用示例

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
        this->AICore().AddConfig(**"**soc_version**"**, aicConfig);
    }
};
```
