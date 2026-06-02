# ValueDepend<a name="ZH-CN_TOPIC_0000002554344739"></a>

## 功能说明<a name="zh-cn_topic_0000001626329929_zh-cn_topic_0000001576875005_zh-cn_topic_0000001525424352_section36583473819"></a>

标识该输入是否为“数据依赖输入”，数据依赖输入是指在Tiling/InferShape等函数实现时依赖该输入的具体数据。该输入数据为host侧数据，开发者在Tiling函数/InferShape函数中可以通过TilingContext类的GetInputTensor/InferShapeContext类的GetInputTensor获取这个输入数据。

## 函数原型<a name="zh-cn_topic_0000001626329929_zh-cn_topic_0000001576875005_zh-cn_topic_0000001525424352_section13230182415108"></a>

```
OpParamDef &ValueDepend(Option value_depend)
OpParamDef &ValueDepend(Option value_depend, DependScope scope)
```

## 参数说明<a name="zh-cn_topic_0000001626329929_zh-cn_topic_0000001576875005_zh-cn_topic_0000001525424352_section75395119104"></a>

<a name="zh-cn_topic_0000001626329929_zh-cn_topic_0000001576875005_zh-cn_topic_0000001575944081_table111938719446"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001626329929_zh-cn_topic_0000001576875005_zh-cn_topic_0000001575944081_row6223476444"><th class="cellrowborder" valign="top" width="17.22%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001626329929_zh-cn_topic_0000001576875005_zh-cn_topic_0000001575944081_p10223674448"><a name="zh-cn_topic_0000001626329929_zh-cn_topic_0000001576875005_zh-cn_topic_0000001575944081_p10223674448"></a><a name="zh-cn_topic_0000001626329929_zh-cn_topic_0000001576875005_zh-cn_topic_0000001575944081_p10223674448"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="15.340000000000002%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001626329929_zh-cn_topic_0000001576875005_zh-cn_topic_0000001575944081_p645511218169"><a name="zh-cn_topic_0000001626329929_zh-cn_topic_0000001576875005_zh-cn_topic_0000001575944081_p645511218169"></a><a name="zh-cn_topic_0000001626329929_zh-cn_topic_0000001576875005_zh-cn_topic_0000001575944081_p645511218169"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="67.44%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001626329929_zh-cn_topic_0000001576875005_zh-cn_topic_0000001575944081_p1922337124411"><a name="zh-cn_topic_0000001626329929_zh-cn_topic_0000001576875005_zh-cn_topic_0000001575944081_p1922337124411"></a><a name="zh-cn_topic_0000001626329929_zh-cn_topic_0000001576875005_zh-cn_topic_0000001575944081_p1922337124411"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001626329929_zh-cn_topic_0000001576875005_zh-cn_topic_0000001575944081_row152234713443"><td class="cellrowborder" valign="top" width="17.22%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001626329929_zh-cn_topic_0000001576875005_zh-cn_topic_0000001575944081_p318615392613"><a name="zh-cn_topic_0000001626329929_zh-cn_topic_0000001576875005_zh-cn_topic_0000001575944081_p318615392613"></a><a name="zh-cn_topic_0000001626329929_zh-cn_topic_0000001576875005_zh-cn_topic_0000001575944081_p318615392613"></a>value_depend</p>
</td>
<td class="cellrowborder" valign="top" width="15.340000000000002%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001626329929_zh-cn_topic_0000001576875005_zh-cn_topic_0000001575944081_p320343694214"><a name="zh-cn_topic_0000001626329929_zh-cn_topic_0000001576875005_zh-cn_topic_0000001575944081_p320343694214"></a><a name="zh-cn_topic_0000001626329929_zh-cn_topic_0000001576875005_zh-cn_topic_0000001575944081_p320343694214"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.44%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001626329929_p15488205233414"><a name="zh-cn_topic_0000001626329929_p15488205233414"></a><a name="zh-cn_topic_0000001626329929_p15488205233414"></a>value_depend有以下两种取值：</p>
<a name="zh-cn_topic_0000001626329929_ul1917131251512"></a><a name="zh-cn_topic_0000001626329929_ul1917131251512"></a><ul id="zh-cn_topic_0000001626329929_ul1917131251512"><li>REQUIRED：表示算子的输入必须是Const类型。<p id="zh-cn_topic_0000001626329929_p67474544184"><a name="zh-cn_topic_0000001626329929_p67474544184"></a><a name="zh-cn_topic_0000001626329929_p67474544184"></a>在调用算子的<a href="SetCheckSupport.md">SetCheckSupport</a>时，会校验算子的输入是否是Const类型。若校验通过，则将此输入的值下发到算子；否则报错。</p>
</li><li>OPTIONAL：表示算子的输入可以是Const类型，也可以不是Const类型。如果输入是Const类型，则将输入的值下发到算子，否则不下发。</li></ul>
</td>
</tr>
<tr id="zh-cn_topic_0000001626329929_row03533135010"><td class="cellrowborder" valign="top" width="17.22%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001626329929_p11333395017"><a name="zh-cn_topic_0000001626329929_p11333395017"></a><a name="zh-cn_topic_0000001626329929_p11333395017"></a>scope</p>
</td>
<td class="cellrowborder" valign="top" width="15.340000000000002%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001626329929_p231133115014"><a name="zh-cn_topic_0000001626329929_p231133115014"></a><a name="zh-cn_topic_0000001626329929_p231133115014"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.44%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001626329929_p1731033135017"><a name="zh-cn_topic_0000001626329929_p1731033135017"></a><a name="zh-cn_topic_0000001626329929_p1731033135017"></a>scope类型为枚举类型DependScope，支持的取值为：</p>
<a name="zh-cn_topic_0000001626329929_ul1782019075220"></a><a name="zh-cn_topic_0000001626329929_ul1782019075220"></a><ul id="zh-cn_topic_0000001626329929_ul1782019075220"><li>ALL：指在Tiling/InferShape等函数实现时都依赖该输入的具体数据，行为与调用单参数ValueDepend重载接口一致。</li><li>TILING：指仅在Tiling时依赖Tensor的值，可以支持Tiling下沉。</li></ul>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0000001626329929_zh-cn_topic_0000001576875005_zh-cn_topic_0000001525424352_section25791320141317"></a>

OpParamDef算子定义，OpParamDef请参考[OpParamDef](OpParamDef.md)。

## 约束说明<a name="zh-cn_topic_0000001626329929_zh-cn_topic_0000001576875005_zh-cn_topic_0000001525424352_section19165124931511"></a>

仅支持对算子输入配置，输入的[参数数据类型](DataType.md)可以配置为DT\_FLOAT/DT\_BOOL/DT\_INT64/DT\_UINT64/DT\_INT32/DT\_UINT32/DT\_INT16/DT\_UINT16/DT\_INT8/DT\_UINT8，且必须满足以下三种情况之一：

1. 输入的[参数数据类型](DataType.md)配置全为DT\_FLOAT，对应生成的输出类型aclFloatArray（aclnn数据类型）。

2. 输入的[参数数据类型](DataType.md)配置全为DT\_BOOL，对应生成的输出类型aclBoolArray（aclnn数据类型）。

3. 输入的[参数数据类型](DataType.md)配置全为整数类型，即DT\_INT64/DT\_UINT64/DT\_INT32/DT\_UINT32/DT\_INT16/DT\_UINT16/DT\_INT8/DT\_UINT8，对应生成的输出类型aclIntArray（aclnn数据类型）。当数据类型配置含有DT\_INT64以外的数据类型时，需要增加一组DT\_INT64对应的输入/输出数据类型组合。

