# DataTypeList<a name="ZH-CN_TOPIC_0000002523304812"></a>

## 功能说明<a name="zh-cn_topic_0000001991854801_zh-cn_topic_0000001526115138_zh-cn_topic_0000001525424352_section36583473819"></a>

定义算子参数数据类型。如果某个输入/输出支持的数据类型支持和其他所有输入/输出支持的数据类型、数据格式组合使用，可以使用该接口定义数据类型。

使用[DataType](DataType.md)配置数据类型时，算子参数的数据类型和格式必须通过显式组合配置，每个组合包含完整的输入/输出数据类型与数据格式的对应关系。如下的示例中表示：当输入x和y数据类型为DT\_FLOAT16时，对应的输出z数据类型也为DT\_FLOAT16，支持的数据格式要求为FORMAT\_ND。

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

如果某个输入/输出支持的数据类型支持和其他所有输入/输出支持的数据类型、数据格式组合使用，使用DataType接口需要写成如下的格式，表示当输入x为DT\_FLOAT16时，支持输入y和输入z的所有数据类型、数据格式组合。

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

## 函数原型<a name="zh-cn_topic_0000001991854801_zh-cn_topic_0000001526115138_zh-cn_topic_0000001525424352_section13230182415108"></a>

```
OpParamDef &DataTypeList(std::vector<ge::DataType> types)
```

## 参数说明<a name="zh-cn_topic_0000001991854801_zh-cn_topic_0000001526115138_zh-cn_topic_0000001525424352_section75395119104"></a>

<a name="zh-cn_topic_0000001991854801_zh-cn_topic_0000001526115138_zh-cn_topic_0000001575944081_table111938719446"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001991854801_zh-cn_topic_0000001526115138_zh-cn_topic_0000001575944081_row6223476444"><th class="cellrowborder" valign="top" width="17.22%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001991854801_zh-cn_topic_0000001526115138_zh-cn_topic_0000001575944081_p10223674448"><a name="zh-cn_topic_0000001991854801_zh-cn_topic_0000001526115138_zh-cn_topic_0000001575944081_p10223674448"></a><a name="zh-cn_topic_0000001991854801_zh-cn_topic_0000001526115138_zh-cn_topic_0000001575944081_p10223674448"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="15.340000000000002%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001991854801_zh-cn_topic_0000001526115138_zh-cn_topic_0000001575944081_p645511218169"><a name="zh-cn_topic_0000001991854801_zh-cn_topic_0000001526115138_zh-cn_topic_0000001575944081_p645511218169"></a><a name="zh-cn_topic_0000001991854801_zh-cn_topic_0000001526115138_zh-cn_topic_0000001575944081_p645511218169"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="67.44%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001991854801_zh-cn_topic_0000001526115138_zh-cn_topic_0000001575944081_p1922337124411"><a name="zh-cn_topic_0000001991854801_zh-cn_topic_0000001526115138_zh-cn_topic_0000001575944081_p1922337124411"></a><a name="zh-cn_topic_0000001991854801_zh-cn_topic_0000001526115138_zh-cn_topic_0000001575944081_p1922337124411"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001991854801_zh-cn_topic_0000001526115138_zh-cn_topic_0000001575944081_row152234713443"><td class="cellrowborder" valign="top" width="17.22%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001991854801_zh-cn_topic_0000001526115138_zh-cn_topic_0000001575944081_p318615392613"><a name="zh-cn_topic_0000001991854801_zh-cn_topic_0000001526115138_zh-cn_topic_0000001575944081_p318615392613"></a><a name="zh-cn_topic_0000001991854801_zh-cn_topic_0000001526115138_zh-cn_topic_0000001575944081_p318615392613"></a><strong id="zh-cn_topic_0000001991854801_b1871392631720"><a name="zh-cn_topic_0000001991854801_b1871392631720"></a><a name="zh-cn_topic_0000001991854801_b1871392631720"></a>types</strong></p>
</td>
<td class="cellrowborder" valign="top" width="15.340000000000002%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001991854801_zh-cn_topic_0000001526115138_zh-cn_topic_0000001575944081_p320343694214"><a name="zh-cn_topic_0000001991854801_zh-cn_topic_0000001526115138_zh-cn_topic_0000001575944081_p320343694214"></a><a name="zh-cn_topic_0000001991854801_zh-cn_topic_0000001526115138_zh-cn_topic_0000001575944081_p320343694214"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.44%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001991854801_zh-cn_topic_0000001526115138_p096733515614"><a name="zh-cn_topic_0000001991854801_zh-cn_topic_0000001526115138_p096733515614"></a><a name="zh-cn_topic_0000001991854801_zh-cn_topic_0000001526115138_p096733515614"></a>算子参数数据类型。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0000001991854801_zh-cn_topic_0000001526115138_zh-cn_topic_0000001525424352_section25791320141317"></a>

OpParamDef算子定义，OpParamDef请参考[OpParamDef](OpParamDef.md)。

## 约束说明<a name="zh-cn_topic_0000001991854801_zh-cn_topic_0000001526115138_zh-cn_topic_0000001525424352_section19165124931511"></a>

-   同一输入/输出不能同时设置DataType和DataTypeList。
-   本接口不支持和[UnknownShapeFormat](UnknownShapeFormat（废弃）.md)同时使用。

## 调用示例<a name="zh-cn_topic_0000001991854801_zh-cn_topic_0000001526115138_zh-cn_topic_0000001575944081_section320753512363"></a>

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

