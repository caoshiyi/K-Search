# HcclServerType

**页面ID:** atlasascendc_api_07_00149  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_00149.html

---

#### 功能说明

配置HCCL的服务端类型。

#### 函数原型

```
void HcclServerType(enum HcclServerType type, const char *soc=nullptr)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| HCCL的服务端类型，类型为HcclServerType枚举类，定义如下：                                                                                                                           ``` namespace ops{ enum HcclServerType : uint32_t {     AICPU = 0,  // AI CPU服务端     AICORE, // AI Core服务端     CCU,    // CCU服务端，仅在昇腾AI处理器包含CCU单元时支持     MAX     // 预留参数，不支持使用 }; } ``` |  |  |
| soc | 输入 | 昇腾AI处理器型号。为该型号配置服务端类型。          可选参数，nullptr或者""表示为算子支持的所有型号配置服务端类型。          soc取值需确保在算子支持的昇腾AI处理器型号范围内，即已经调用AddConfig接口注册。          填写规则请参考算子工程目录下编译配置项文件CMakePresets.json中的ASCEND_COMPUTE_UNIT字段，该字段取值在使用msOpGen创建工程时自动生成。 |

#### 约束说明

- 使用该接口前，算子需要先通过MC2接口注册该算子是通算融合算子，注册后即返回一个OpMC2Def结构。
- 同时为特定昇腾AI处理器型号和所有昇腾AI处理器型号配置服务端类型时，特定昇腾AI处理器型号配置的优先级更高。

#### 调用示例

```
class MC2Custom : public OpDef {
public:
    MC2Custom(const char* name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("z").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Attr("group").AttrType(REQUIRED).String();
        this->AICore().AddConfig("ascendxxx1");
        this->AICore().AddConfig("ascendxxx2");
        this->MC2().HcclGroup("group"); // 配置通信域名称为group
        this->MC2().HcclServerType(HcclServerType::AICPU, "ascendxxx1"); // 配置ascendxxx1型号的通信模式为AI CPU
        this->MC2().HcclServerType(HcclServerType::AICORE); // 配置其他型号即ascendxxx2的通信模式为AI Core
    }
};
OP_ADD(MC2Custom);
```
