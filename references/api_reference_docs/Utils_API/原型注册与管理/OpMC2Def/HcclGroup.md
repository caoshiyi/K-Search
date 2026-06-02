# HcclGroup

**页面ID:** atlasascendc_api_07_1002  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_1002.html

---

#### 功能说明

配置通信域名称，每个名称对应一个通信域。配置后在Kernel侧调用GetHcclContext接口可获取通信域对应的context（消息区）地址。

#### 函数原型

```
OpMC2Def &HcclGroup(const char *value)
OpMC2Def &HcclGroup(std::vector<const char *> value)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| value | 输入 | 配置的通信域名称。单个通信域使用const char *，多通信域使用std::vector<const char *>。 |

#### 约束说明

使用该接口前，算子需要先通过MC2接口注册该算子是通算融合算子，注册后即返回一个OpMC2Def结构。

通信域名称必须先配置为REQUIRED String类型的属性，属性名即为通信域名称。

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
        this->AICore().AddConfig("ascendxxx");
        this->MC2().HcclGroup("group"); // 配置通信域名称为group
    }
};
OP_ADD(MC2Custom);
```
