# FormatList

**页面ID:** atlasascendc_api_07_0962  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0962.html

---

#### 功能说明

定义算子参数数据格式。如果某个输入/输出支持的数据格式支持和其他所有输入/输出支持的数据类型、数据格式组合使用，可以使用该接口定义数据格式。

使用Format配置数据格式时，这些数据格式和其他输入输出的数据类型和数据格式是一一对应的，如下的示例中表示：当输入x和y数据格式为FORMAT_NHWC时，对应的输出z数据格式也为FORMAT_NHWC，且此时x、y、z的数据类型要求为ge::DT_FLOAT。

```
class AddCustom : public OpDef {
public:
    AddCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_NHWC, ge::FORMAT_ND});
        this->Input("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_NHWC, ge::FORMAT_ND});
        this->Output("z")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_NHWC, ge::FORMAT_ND});
        ...
    }
};
```

如果某个输入/输出支持的数据格式支持和其他所有输入/输出支持的数据类型、数据格式组合使用，使用Format接口需要写成如下的格式，表示当输入x为FORMAT_ND时，支持输入y和输入z的所有数据类型、数据格式组合。

```
class XxxCustom : public OpDef {
public:
    XxxCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("z")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        ...
    }
};
```

此时可以通过FormatList指定数据类型，无需重复列出，例如：

```
class XxxCustom : public OpDef {
public:
    XxxCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND});
        this->Input("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("z")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        ...
    }
};
```

#### 函数原型

```
OpParamDef &FormatList(std::vector<ge::Format> formats)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| formats | 输入 | 算子参数数据格式，ge::Format请参考Format。 |

#### 返回值说明

OpParamDef算子定义，OpParamDef请参考OpParamDef。

#### 约束说明

- 同一输入/输出不能同时设置Format和FormatList。
- 本接口不支持和UnknownShapeFormat同时使用。

#### 调用示例

```
class AddCustom : public OpDef {
public:
    AddCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32})
            .FormatList({ge::FORMAT_ND});
        this->Input("x1")
        ......
    }
};
```
