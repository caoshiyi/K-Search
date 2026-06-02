# OpAICoreConfig构造函数

**页面ID:** atlasascendc_api_07_00139  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_00139.html

---

#### 功能说明

OpAICoreConfig构造函数。

#### 函数原型

```
OpAICoreConfig()
OpAICoreConfig(const char *soc)
```

#### 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| soc | 输入 | AI处理器型号。 |

#### 约束说明

传入soc入参的构造函数会对OpAICoreConfig结构中的部分参数进行初始化，具体的参数和初始化值如下表所示：

| 配置参数 | 说明 | 初始化值 |
| --- | --- | --- |
| DynamicCompileStaticFlag | 用于标识该算子实现是否支持入图时的静态Shape编译。 | true |
| DynamicFormatFlag | 标识是否根据SetOpSelectFormat设置的函数自动推导算子输入输出支持的dtype和format。 | true |
| DynamicRankSupportFlag | 标识算子是否支持dynamicRank（动态维度）。 | true |
| DynamicShapeSupportFlag | 用于标识该算子是否支持入图时的动态Shape场景。 | true |
| NeedCheckSupportFlag | 标识是否在算子融合阶段调用算子参数校验函数进行data type与Shape的校验。 | false |
| PrecisionReduceFlag | 此字段用于进行ATC模型转换或者进行网络调测时，控制算子的精度模式。 | true |

无入参的默认构造函数不会初始化上述参数。

#### 调用示例

```
class AddCustom : public OpDef {
public:
    AddCustom(const char* name) : OpDef(name)
    {
        this->Input("x").DataType({ ge::DT_FLOAT16 }).ParamType(OPTIONAL);
        this->Output("y").DataType({ ge::DT_FLOAT16 });
        // 使用soc入参构造函数
        OpAICoreConfig aicConfig1("ascendxxx1");
        OpAICoreConfig aicConfig2("ascendxxx2");
        aicConfig1.Input("x")
            .ParamType(OPTIONAL)
            .DataType({ ge::DT_FLOAT })
            .Format({ ge::FORMAT_ND });
        aicConfig2.Input("x")
            .ParamType(REQUIRED)
            .DataType({ ge::DT_INT32 })
            .Format({ ge::FORMAT_ND });
        this->AICore().AddConfig("ascendxxx1", aicConfig1);
        this->AICore().AddConfig("ascendxxx2", aicConfig2);
    }
};
```
