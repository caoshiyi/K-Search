# Output

**页面ID:** atlasascendc_api_07_0990  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0990.html

---

#### 功能说明

某些场景下，同一个算子在不同的AI处理器型号上，其支持的原型输出不同。

通过该接口，可针对不同的AI处理器型号注册差异化的算子输出。调用该接口后会返回一个OpParamDef结构，后续可通过该结构配置算子输出信息。

#### 函数原型

```
OpParamDef &Output(const char *name)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| name | 输入 | 算子输出名称。 |

#### 返回值说明

算子参数定义，OpParamDef请参考OpParamDef。

#### 约束说明

无

#### 调用示例

```
class AddCustom : public OpDef {
public:
    AddCustom(const char* name) : OpDef(name)
    {
        this->Input("x").DataType({ ge::DT_FLOAT16 }).ParamType(OPTIONAL);
        this->Output("y").DataType({ ge::DT_FLOAT16 });
        OpAICoreConfig aicConfig1;
        OpAICoreConfig aicConfig2;
        aicConfig1.Output("y")
            .ParamType(OPTIONAL)
            .DataType({ ge::DT_FLOAT })
            .Format({ ge::FORMAT_ND });
        aicConfig2.Output("y")
            .ParamType(REQUIRED)
            .DataType({ ge::DT_INT32 })
            .Format({ ge::FORMAT_ND });
        this->AICore().AddConfig("ascendxxx1", aicConfig1);
        this->AICore().AddConfig("ascendxxx2", aicConfig2);
    }
};
```
