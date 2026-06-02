# DataTypeList

**页面ID:** atlasascendc_api_07_0960  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0960.html

---

#### 功能说明

定义算子参数数据类型。如果某个输入/输出支持的数据类型支持和其他所有输入/输出支持的数据类型、数据格式组合使用，可以使用该接口定义数据类型。

使用DataType配置数据类型时，算子参数的数据类型和格式必须通过显式组合配置，每个组合包含完整的输入/输出数据类型与数据格式的对应关系。如下的示例中表示：当输入x和y数据类型为DT_FLOAT16时，对应的输出z数据类型也为DT_FLOAT16，支持的数据格式要求为FORMAT_ND。

```
class AddCustom : public OpDef {
public:
    AddCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32})
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

如果某个输入/输出支持的数据类型支持和其他所有输入/输出支持的数据类型、数据格式组合使用，使用DataType接口需要写成如下的格式，表示当输入x为DT_FLOAT16时，支持输入y和输入z的所有数据类型、数据格式组合。

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

此时可以通过DataTypeList指定数据类型，无需重复列出，例如：

```
class XxxCustom : public OpDef {
public:
    XxxCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_FLOAT16})
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

#### 函数原型

```
OpParamDef &DataTypeList(std::vector<ge::DataType> types)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| **types** | 输入 | 算子参数数据类型，ge::DataType请参考DataType。 |

#### 返回值说明

OpParamDef算子定义，OpParamDef请参考OpParamDef。

#### 约束说明

- 同一输入/输出不能同时设置DataType和DataTypeList。
- 本接口不支持和UnknownShapeFormat同时使用。

#### 调用示例

```
class AddCustom : public OpDef {
public:
    AddCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_NCHW});
        this->Input("x1")
             ......
    }
};
```
