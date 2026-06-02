# SetInferDataType<a name="ZH-CN_TOPIC_0000002523344196"></a>

## 功能说明<a name="zh-cn_topic_0000001549667388_zh-cn_topic_0000001526432182_zh-cn_topic_0000001575944081_section36583473819"></a>

使用图模式时，需要调用该接口注册DataType推导函数。

## 函数原型<a name="zh-cn_topic_0000001549667388_zh-cn_topic_0000001526432182_zh-cn_topic_0000001575944081_section13230182415108"></a>

```
OpDef &SetInferDataType(gert::OpImplRegisterV2::InferDataTypeKernelFunc func)
```

## 参数说明<a name="zh-cn_topic_0000001549667388_zh-cn_topic_0000001526432182_zh-cn_topic_0000001575944081_section75395119104"></a>

<a name="zh-cn_topic_0000001549667388_zh-cn_topic_0000001526432182_zh-cn_topic_0000001575944081_table111938719446"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001549667388_zh-cn_topic_0000001526432182_zh-cn_topic_0000001575944081_row6223476444"><th class="cellrowborder" valign="top" width="17.22%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001549667388_zh-cn_topic_0000001526432182_zh-cn_topic_0000001575944081_p10223674448"><a name="zh-cn_topic_0000001549667388_zh-cn_topic_0000001526432182_zh-cn_topic_0000001575944081_p10223674448"></a><a name="zh-cn_topic_0000001549667388_zh-cn_topic_0000001526432182_zh-cn_topic_0000001575944081_p10223674448"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="15.340000000000002%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001549667388_zh-cn_topic_0000001526432182_zh-cn_topic_0000001575944081_p645511218169"><a name="zh-cn_topic_0000001549667388_zh-cn_topic_0000001526432182_zh-cn_topic_0000001575944081_p645511218169"></a><a name="zh-cn_topic_0000001549667388_zh-cn_topic_0000001526432182_zh-cn_topic_0000001575944081_p645511218169"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="67.44%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001549667388_zh-cn_topic_0000001526432182_zh-cn_topic_0000001575944081_p1922337124411"><a name="zh-cn_topic_0000001549667388_zh-cn_topic_0000001526432182_zh-cn_topic_0000001575944081_p1922337124411"></a><a name="zh-cn_topic_0000001549667388_zh-cn_topic_0000001526432182_zh-cn_topic_0000001575944081_p1922337124411"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001549667388_zh-cn_topic_0000001526432182_zh-cn_topic_0000001575944081_row152234713443"><td class="cellrowborder" valign="top" width="17.22%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001549667388_zh-cn_topic_0000001526432182_p243146152017"><a name="zh-cn_topic_0000001549667388_zh-cn_topic_0000001526432182_p243146152017"></a><a name="zh-cn_topic_0000001549667388_zh-cn_topic_0000001526432182_p243146152017"></a>func</p>
</td>
<td class="cellrowborder" valign="top" width="15.340000000000002%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001549667388_zh-cn_topic_0000001526432182_zh-cn_topic_0000001575944081_p320343694214"><a name="zh-cn_topic_0000001549667388_zh-cn_topic_0000001526432182_zh-cn_topic_0000001575944081_p320343694214"></a><a name="zh-cn_topic_0000001549667388_zh-cn_topic_0000001526432182_zh-cn_topic_0000001575944081_p320343694214"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.44%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001549667388_zh-cn_topic_0000001526432182_p155137179184"><a name="zh-cn_topic_0000001549667388_zh-cn_topic_0000001526432182_p155137179184"></a><a name="zh-cn_topic_0000001549667388_zh-cn_topic_0000001526432182_p155137179184"></a>DataType推导函数。<strong id="zh-cn_topic_0000001549667388_zh-cn_topic_0000001526432182_b15918130102019"><a name="zh-cn_topic_0000001549667388_zh-cn_topic_0000001526432182_b15918130102019"></a><a name="zh-cn_topic_0000001549667388_zh-cn_topic_0000001526432182_b15918130102019"></a>InferDataTypeKernelFunc</strong>类型定义如下：</p>
<a name="zh-cn_topic_0000001549667388_zh-cn_topic_0000001526432182_screen17603182541315"></a><a name="zh-cn_topic_0000001549667388_zh-cn_topic_0000001526432182_screen17603182541315"></a><pre class="screen" codetype="Cpp" id="zh-cn_topic_0000001549667388_zh-cn_topic_0000001526432182_screen17603182541315">using InferDataTypeKernelFunc = UINT32 (*)(InferDataTypeContext *);</pre>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0000001549667388_zh-cn_topic_0000001526432182_zh-cn_topic_0000001575944081_section25791320141317"></a>

OpDef算子定义，OpDef请参考[OpDef](OpDef.md)。

## 约束说明<a name="zh-cn_topic_0000001549667388_zh-cn_topic_0000001526432182_zh-cn_topic_0000001575944081_section19165124931511"></a>

无

