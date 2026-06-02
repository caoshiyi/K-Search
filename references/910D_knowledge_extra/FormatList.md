# FormatList<a name="ZH-CN_TOPIC_0000002554424411"></a>

## 功能说明<a name="zh-cn_topic_0000001991734609_zh-cn_topic_0000001526275046_zh-cn_topic_0000001525424352_section36583473819"></a>

定义算子参数数据格式。如果某个输入/输出支持的数据格式支持和其他所有输入/输出支持的数据类型、数据格式组合使用，可以使用该接口定义数据格式。

使用[Format](Format.md)配置数据格式时，这些数据格式和其他输入输出的数据类型和数据格式是一一对应的，如下的示例中表示：当输入x和y数据格式为FORMAT\_NHWC时，对应的输出z数据格式也为FORMAT\_NHWC，且此时x、y、z的数据类型要求为ge::DT\_FLOAT。

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

如果某个输入/输出支持的数据格式支持和其他所有输入/输出支持的数据类型、数据格式组合使用，使用Format接口需要写成如下的格式，表示当输入x为FORMAT\_ND时，支持输入y和输入z的所有数据类型、数据格式组合。

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

## 函数原型<a name="zh-cn_topic_0000001991734609_zh-cn_topic_0000001526275046_zh-cn_topic_0000001525424352_section13230182415108"></a>

```
OpParamDef &FormatList(std::vector<ge::Format> formats)
```

## 参数说明<a name="zh-cn_topic_0000001991734609_zh-cn_topic_0000001526275046_zh-cn_topic_0000001525424352_section75395119104"></a>

<a name="zh-cn_topic_0000001991734609_zh-cn_topic_0000001526275046_zh-cn_topic_0000001575944081_table111938719446"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001991734609_zh-cn_topic_0000001526275046_zh-cn_topic_0000001575944081_row6223476444"><th class="cellrowborder" valign="top" width="17.22%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001991734609_zh-cn_topic_0000001526275046_zh-cn_topic_0000001575944081_p10223674448"><a name="zh-cn_topic_0000001991734609_zh-cn_topic_0000001526275046_zh-cn_topic_0000001575944081_p10223674448"></a><a name="zh-cn_topic_0000001991734609_zh-cn_topic_0000001526275046_zh-cn_topic_0000001575944081_p10223674448"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="15.340000000000002%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001991734609_zh-cn_topic_0000001526275046_zh-cn_topic_0000001575944081_p645511218169"><a name="zh-cn_topic_0000001991734609_zh-cn_topic_0000001526275046_zh-cn_topic_0000001575944081_p645511218169"></a><a name="zh-cn_topic_0000001991734609_zh-cn_topic_0000001526275046_zh-cn_topic_0000001575944081_p645511218169"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="67.44%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001991734609_zh-cn_topic_0000001526275046_zh-cn_topic_0000001575944081_p1922337124411"><a name="zh-cn_topic_0000001991734609_zh-cn_topic_0000001526275046_zh-cn_topic_0000001575944081_p1922337124411"></a><a name="zh-cn_topic_0000001991734609_zh-cn_topic_0000001526275046_zh-cn_topic_0000001575944081_p1922337124411"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001991734609_zh-cn_topic_0000001526275046_zh-cn_topic_0000001575944081_row152234713443"><td class="cellrowborder" valign="top" width="17.22%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001991734609_zh-cn_topic_0000001526275046_zh-cn_topic_0000001575944081_p318615392613"><a name="zh-cn_topic_0000001991734609_zh-cn_topic_0000001526275046_zh-cn_topic_0000001575944081_p318615392613"></a><a name="zh-cn_topic_0000001991734609_zh-cn_topic_0000001526275046_zh-cn_topic_0000001575944081_p318615392613"></a>formats</p>
</td>
<td class="cellrowborder" valign="top" width="15.340000000000002%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001991734609_zh-cn_topic_0000001526275046_zh-cn_topic_0000001575944081_p320343694214"><a name="zh-cn_topic_0000001991734609_zh-cn_topic_0000001526275046_zh-cn_topic_0000001575944081_p320343694214"></a><a name="zh-cn_topic_0000001991734609_zh-cn_topic_0000001526275046_zh-cn_topic_0000001575944081_p320343694214"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.44%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001991734609_zh-cn_topic_0000001526275046_p43231148103313"><a name="zh-cn_topic_0000001991734609_zh-cn_topic_0000001526275046_p43231148103313"></a><a name="zh-cn_topic_0000001991734609_zh-cn_topic_0000001526275046_p43231148103313"></a>算子参数数据格式。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0000001991734609_zh-cn_topic_0000001526275046_zh-cn_topic_0000001525424352_section25791320141317"></a>

OpParamDef算子定义，OpParamDef请参考[OpParamDef](OpParamDef.md)。

## 约束说明<a name="zh-cn_topic_0000001991734609_zh-cn_topic_0000001526275046_zh-cn_topic_0000001525424352_section19165124931511"></a>

-   同一输入/输出不能同时设置Format和FormatList。
-   本接口不支持和[UnknownShapeFormat](UnknownShapeFormat（废弃）.md)同时使用。

## 调用示例<a name="zh-cn_topic_0000001991734609_zh-cn_topic_0000001526275046_zh-cn_topic_0000001575944081_section320753512363"></a>

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

