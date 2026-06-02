# SetTiling<a name="ZH-CN_TOPIC_0000002554344073"></a>

## 功能说明<a name="zh-cn_topic_0000001549507408_zh-cn_topic_0000001576923281_zh-cn_topic_0000001525424352_section36583473819"></a>

注册Tiling函数。Tiling函数的原型是固定的，接受一个TilingContext作为输入，在此context上可以获取到输入、输出的Shape指针等内容。注册的Tiling函数由框架调用，调用时会传入TilingContext参数。

## 函数原型<a name="zh-cn_topic_0000001549507408_zh-cn_topic_0000001576923281_zh-cn_topic_0000001525424352_section13230182415108"></a>

```
OpAICoreDef &SetTiling(gert::OpImplRegisterV2::TilingKernelFunc func)
```

## 参数说明<a name="zh-cn_topic_0000001549507408_zh-cn_topic_0000001576923281_zh-cn_topic_0000001525424352_section75395119104"></a>

<a name="zh-cn_topic_0000001549507408_zh-cn_topic_0000001576923281_zh-cn_topic_0000001575944081_table111938719446"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001549507408_zh-cn_topic_0000001576923281_zh-cn_topic_0000001575944081_row6223476444"><th class="cellrowborder" valign="top" width="17.22%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001549507408_zh-cn_topic_0000001576923281_zh-cn_topic_0000001575944081_p10223674448"><a name="zh-cn_topic_0000001549507408_zh-cn_topic_0000001576923281_zh-cn_topic_0000001575944081_p10223674448"></a><a name="zh-cn_topic_0000001549507408_zh-cn_topic_0000001576923281_zh-cn_topic_0000001575944081_p10223674448"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="15.340000000000002%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001549507408_zh-cn_topic_0000001576923281_zh-cn_topic_0000001575944081_p645511218169"><a name="zh-cn_topic_0000001549507408_zh-cn_topic_0000001576923281_zh-cn_topic_0000001575944081_p645511218169"></a><a name="zh-cn_topic_0000001549507408_zh-cn_topic_0000001576923281_zh-cn_topic_0000001575944081_p645511218169"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="67.44%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001549507408_zh-cn_topic_0000001576923281_zh-cn_topic_0000001575944081_p1922337124411"><a name="zh-cn_topic_0000001549507408_zh-cn_topic_0000001576923281_zh-cn_topic_0000001575944081_p1922337124411"></a><a name="zh-cn_topic_0000001549507408_zh-cn_topic_0000001576923281_zh-cn_topic_0000001575944081_p1922337124411"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001549507408_zh-cn_topic_0000001576923281_zh-cn_topic_0000001575944081_row152234713443"><td class="cellrowborder" valign="top" width="17.22%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001549507408_zh-cn_topic_0000001576923281_zh-cn_topic_0000001575944081_p318615392613"><a name="zh-cn_topic_0000001549507408_zh-cn_topic_0000001576923281_zh-cn_topic_0000001575944081_p318615392613"></a><a name="zh-cn_topic_0000001549507408_zh-cn_topic_0000001576923281_zh-cn_topic_0000001575944081_p318615392613"></a>func</p>
</td>
<td class="cellrowborder" valign="top" width="15.340000000000002%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001549507408_zh-cn_topic_0000001576923281_zh-cn_topic_0000001575944081_p320343694214"><a name="zh-cn_topic_0000001549507408_zh-cn_topic_0000001576923281_zh-cn_topic_0000001575944081_p320343694214"></a><a name="zh-cn_topic_0000001549507408_zh-cn_topic_0000001576923281_zh-cn_topic_0000001575944081_p320343694214"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.44%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001549507408_zh-cn_topic_0000001576923281_p12935163055011"><a name="zh-cn_topic_0000001549507408_zh-cn_topic_0000001576923281_p12935163055011"></a><a name="zh-cn_topic_0000001549507408_zh-cn_topic_0000001576923281_p12935163055011"></a>Tiling函数。TilingKernelFunc类型定义如下：</p>
<a name="zh-cn_topic_0000001549507408_zh-cn_topic_0000001576923281_screen2936530125010"></a><a name="zh-cn_topic_0000001549507408_zh-cn_topic_0000001576923281_screen2936530125010"></a><pre class="screen" codetype="Cpp" id="zh-cn_topic_0000001549507408_zh-cn_topic_0000001576923281_screen2936530125010">using TilingKernelFunc = UINT32 (*)(TilingContext *);</pre>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0000001549507408_zh-cn_topic_0000001576923281_zh-cn_topic_0000001525424352_section25791320141317"></a>

OpAICoreDef算子定义，OpAICoreDef请参考[OpAICoreDef](OpAICoreDef.md)。

## 约束说明<a name="zh-cn_topic_0000001549507408_zh-cn_topic_0000001576923281_zh-cn_topic_0000001525424352_section19165124931511"></a>

无

