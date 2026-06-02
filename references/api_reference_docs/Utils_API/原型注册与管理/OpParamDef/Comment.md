# Comment

**页面ID:** atlasascendc_api_07_0975  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0975.html

---

#### 功能说明

设置input/output参数的注释。

#### 函数原型

```
OpParamDef &Comment(const char *comment)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| comment | 输入 | 注释内容。 |

#### 返回值说明

算子参数定义，OpParamDef请参考OpParamDef。

#### 约束说明

无

#### 调用示例

```
class AddCustom : public OpDef {
public:
    explicit AddCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .Comment("Input cmt 1"); // 注释内容
        this->Input("y")
            .ParamType(REQUIRED)
            .Comment("Input cmt 2") // 注释内容
            .DataType({ge::DT_FLOAT, ge::DT_INT32})
            .FormatList({ge::FORMAT_ND});

        this->Output("z")
            .Comment("Output cmt 1") // 注释内容
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_INT32})
            .FormatList({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascendxxx");

    }
};
```
