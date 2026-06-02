# Comment<a name="ZH-CN_TOPIC_0000002554344069"></a>

## 功能说明<a name="zh-cn_topic_0000002055436684_zh-cn_topic_0000001526111046_zh-cn_topic_0000001525424352_section36583473819"></a>

设置input/output参数的注释。用于在自动生成算子原型头文件时，同步生成算子原型注释。

基于OpDef算子原型定义，自定义算子工程可以实现如下自动化能力：自动生成图模式场景使用的算子原型定义REG\_OP（算子原型头文件），开发者可以使用生成的算子原型进行构图、图编译、图执行等操作。

生成的注释有助于辅助理解算子原型，并可以基于这些注释自动生成算子原型的文档说明。通常情况下，内置CANN算子使用较多。开发者可以按需使用。

## 函数原型<a name="zh-cn_topic_0000002055436684_zh-cn_topic_0000001526111046_zh-cn_topic_0000001525424352_section13230182415108"></a>

```
OpParamDef &Comment(const char *comment)
```

## 参数说明<a name="zh-cn_topic_0000002055436684_zh-cn_topic_0000001526111046_zh-cn_topic_0000001525424352_section75395119104"></a>

<a name="zh-cn_topic_0000002055436684_zh-cn_topic_0000001526111046_zh-cn_topic_0000001575944081_table111938719446"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000002055436684_zh-cn_topic_0000001526111046_zh-cn_topic_0000001575944081_row6223476444"><th class="cellrowborder" valign="top" width="17.22%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000002055436684_zh-cn_topic_0000001526111046_zh-cn_topic_0000001575944081_p10223674448"><a name="zh-cn_topic_0000002055436684_zh-cn_topic_0000001526111046_zh-cn_topic_0000001575944081_p10223674448"></a><a name="zh-cn_topic_0000002055436684_zh-cn_topic_0000001526111046_zh-cn_topic_0000001575944081_p10223674448"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="15.340000000000002%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000002055436684_zh-cn_topic_0000001526111046_zh-cn_topic_0000001575944081_p645511218169"><a name="zh-cn_topic_0000002055436684_zh-cn_topic_0000001526111046_zh-cn_topic_0000001575944081_p645511218169"></a><a name="zh-cn_topic_0000002055436684_zh-cn_topic_0000001526111046_zh-cn_topic_0000001575944081_p645511218169"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="67.44%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000002055436684_zh-cn_topic_0000001526111046_zh-cn_topic_0000001575944081_p1922337124411"><a name="zh-cn_topic_0000002055436684_zh-cn_topic_0000001526111046_zh-cn_topic_0000001575944081_p1922337124411"></a><a name="zh-cn_topic_0000002055436684_zh-cn_topic_0000001526111046_zh-cn_topic_0000001575944081_p1922337124411"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000002055436684_zh-cn_topic_0000001526111046_zh-cn_topic_0000001575944081_row152234713443"><td class="cellrowborder" valign="top" width="17.22%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002055436684_zh-cn_topic_0000001526111046_zh-cn_topic_0000001575944081_p318615392613"><a name="zh-cn_topic_0000002055436684_zh-cn_topic_0000001526111046_zh-cn_topic_0000001575944081_p318615392613"></a><a name="zh-cn_topic_0000002055436684_zh-cn_topic_0000001526111046_zh-cn_topic_0000001575944081_p318615392613"></a>comment</p>
</td>
<td class="cellrowborder" valign="top" width="15.340000000000002%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002055436684_zh-cn_topic_0000001526111046_zh-cn_topic_0000001575944081_p320343694214"><a name="zh-cn_topic_0000002055436684_zh-cn_topic_0000001526111046_zh-cn_topic_0000001575944081_p320343694214"></a><a name="zh-cn_topic_0000002055436684_zh-cn_topic_0000001526111046_zh-cn_topic_0000001575944081_p320343694214"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.44%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002055436684_zh-cn_topic_0000001526111046_p096733515614"><a name="zh-cn_topic_0000002055436684_zh-cn_topic_0000001526111046_p096733515614"></a><a name="zh-cn_topic_0000002055436684_zh-cn_topic_0000001526111046_p096733515614"></a>注释内容。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0000002055436684_zh-cn_topic_0000001526111046_zh-cn_topic_0000001525424352_section25791320141317"></a>

算子参数定义，OpParamDef请参考[OpParamDef](OpParamDef.md)。

## 约束说明<a name="zh-cn_topic_0000002055436684_zh-cn_topic_0000001526111046_zh-cn_topic_0000001525424352_section19165124931511"></a>

无

## 调用示例<a name="zh-cn_topic_0000002055436684_zh-cn_topic_0000001526111046_zh-cn_topic_0000001575944081_section320753512363"></a>

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

