# To<a name="ZH-CN_TOPIC_0000002554343865"></a>

## 功能说明<a name="zh-cn_topic_0000001759578869_zh-cn_topic_0000001526594958_zh-cn_topic_0000001525424352_section36583473819"></a>

配置该参数后，在调用单算子API\(aclnnxxx\)接口时，会将aclScalar/aclScalarList的数据类型转换为该参数指定的数据类型。

指定的数据类型可以通过两种方式传入：

-   直接传入datatype；
-   传入输入的名称，表示数据类型和该输入的datatype保持一致。

**该接口仅在如下场景支持：**

-   通过单算子API执行的方式开发单算子调用应用。
-   间接调用单算子API\(aclnnxxx\)接口：Pytorch框架单算子直调的场景。

## 函数原型<a name="zh-cn_topic_0000001759578869_zh-cn_topic_0000001526594958_zh-cn_topic_0000001525424352_section13230182415108"></a>

```
OpParamDef &To(const ge::DataType type)
OpParamDef &To(const char *name)
```

## 参数说明<a name="zh-cn_topic_0000001759578869_zh-cn_topic_0000001526594958_zh-cn_topic_0000001525424352_section75395119104"></a>

<a name="zh-cn_topic_0000001759578869_zh-cn_topic_0000001526594958_zh-cn_topic_0000001575944081_table111938719446"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001759578869_zh-cn_topic_0000001526594958_zh-cn_topic_0000001575944081_row6223476444"><th class="cellrowborder" valign="top" width="17.22%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001759578869_zh-cn_topic_0000001526594958_zh-cn_topic_0000001575944081_p10223674448"><a name="zh-cn_topic_0000001759578869_zh-cn_topic_0000001526594958_zh-cn_topic_0000001575944081_p10223674448"></a><a name="zh-cn_topic_0000001759578869_zh-cn_topic_0000001526594958_zh-cn_topic_0000001575944081_p10223674448"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="15.340000000000002%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001759578869_zh-cn_topic_0000001526594958_zh-cn_topic_0000001575944081_p645511218169"><a name="zh-cn_topic_0000001759578869_zh-cn_topic_0000001526594958_zh-cn_topic_0000001575944081_p645511218169"></a><a name="zh-cn_topic_0000001759578869_zh-cn_topic_0000001526594958_zh-cn_topic_0000001575944081_p645511218169"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="67.44%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001759578869_zh-cn_topic_0000001526594958_zh-cn_topic_0000001575944081_p1922337124411"><a name="zh-cn_topic_0000001759578869_zh-cn_topic_0000001526594958_zh-cn_topic_0000001575944081_p1922337124411"></a><a name="zh-cn_topic_0000001759578869_zh-cn_topic_0000001526594958_zh-cn_topic_0000001575944081_p1922337124411"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001759578869_zh-cn_topic_0000001526594958_zh-cn_topic_0000001575944081_row152234713443"><td class="cellrowborder" valign="top" width="17.22%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001759578869_zh-cn_topic_0000001526594958_zh-cn_topic_0000001575944081_p318615392613"><a name="zh-cn_topic_0000001759578869_zh-cn_topic_0000001526594958_zh-cn_topic_0000001575944081_p318615392613"></a><a name="zh-cn_topic_0000001759578869_zh-cn_topic_0000001526594958_zh-cn_topic_0000001575944081_p318615392613"></a><strong id="zh-cn_topic_0000001759578869_b124780476475"><a name="zh-cn_topic_0000001759578869_b124780476475"></a><a name="zh-cn_topic_0000001759578869_b124780476475"></a>type</strong></p>
</td>
<td class="cellrowborder" valign="top" width="15.340000000000002%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001759578869_zh-cn_topic_0000001526594958_zh-cn_topic_0000001575944081_p320343694214"><a name="zh-cn_topic_0000001759578869_zh-cn_topic_0000001526594958_zh-cn_topic_0000001575944081_p320343694214"></a><a name="zh-cn_topic_0000001759578869_zh-cn_topic_0000001526594958_zh-cn_topic_0000001575944081_p320343694214"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.44%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001759578869_zh-cn_topic_0000001526594958_p096733515614"><a name="zh-cn_topic_0000001759578869_zh-cn_topic_0000001526594958_p096733515614"></a><a name="zh-cn_topic_0000001759578869_zh-cn_topic_0000001526594958_p096733515614"></a>指定的数据类型。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001759578869_row19976124917474"><td class="cellrowborder" valign="top" width="17.22%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001759578869_p12976184904720"><a name="zh-cn_topic_0000001759578869_p12976184904720"></a><a name="zh-cn_topic_0000001759578869_p12976184904720"></a><strong id="zh-cn_topic_0000001759578869_b73001254114718"><a name="zh-cn_topic_0000001759578869_b73001254114718"></a><a name="zh-cn_topic_0000001759578869_b73001254114718"></a>name</strong></p>
</td>
<td class="cellrowborder" valign="top" width="15.340000000000002%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001759578869_p1497694924718"><a name="zh-cn_topic_0000001759578869_p1497694924718"></a><a name="zh-cn_topic_0000001759578869_p1497694924718"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.44%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001759578869_p179771249204713"><a name="zh-cn_topic_0000001759578869_p179771249204713"></a><a name="zh-cn_topic_0000001759578869_p179771249204713"></a>算子输入的名称，表示指定的数据类型和该输入的数据类型一致。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0000001759578869_zh-cn_topic_0000001526594958_zh-cn_topic_0000001525424352_section25791320141317"></a>

OpParamDef算子定义，OpParamDef请参考[OpParamDef](OpParamDef.md)。

## 约束说明<a name="zh-cn_topic_0000001759578869_zh-cn_topic_0000001526594958_zh-cn_topic_0000001525424352_section19165124931511"></a>

无

## 调用示例<a name="zh-cn_topic_0000001759578869_zh-cn_topic_0000001526594958_zh-cn_topic_0000001575944081_section320753512363"></a>

```
this->Input("x")
    .ParamType(REQUIRED)
    .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
    .Format({ge::FORMAT_ND, ge::FORMAT_ND})
    .ScalarList()
    .To(ge::DT_FLOAT);
this->Input("x1")
    .ParamType(REQUIRED)
    .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
    .Format({ge::FORMAT_ND, ge::FORMAT_ND});
this->Input("x2")
    .ParamType(REQUIRED)
    .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
    .Format({ge::FORMAT_ND, ge::FORMAT_ND})
    .ScalarList()
    .To("x1");
```

