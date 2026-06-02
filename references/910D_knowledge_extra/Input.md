# Input<a name="ZH-CN_TOPIC_0000002523344118"></a>

## 功能说明<a name="zh-cn_topic_0000001549188224_zh-cn_topic_0000001526111046_zh-cn_topic_0000001525424352_section36583473819"></a>

注册算子输入，调用该接口后会返回一个OpParamDef结构，后续可通过该结构配置算子输入信息。

## 函数原型<a name="zh-cn_topic_0000001549188224_zh-cn_topic_0000001526111046_zh-cn_topic_0000001525424352_section13230182415108"></a>

```
OpParamDef &Input(const char *name)
```

## 参数说明<a name="zh-cn_topic_0000001549188224_zh-cn_topic_0000001526111046_zh-cn_topic_0000001525424352_section75395119104"></a>

<a name="zh-cn_topic_0000001549188224_zh-cn_topic_0000001526111046_zh-cn_topic_0000001575944081_table111938719446"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001549188224_zh-cn_topic_0000001526111046_zh-cn_topic_0000001575944081_row6223476444"><th class="cellrowborder" valign="top" width="17.22%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001549188224_zh-cn_topic_0000001526111046_zh-cn_topic_0000001575944081_p10223674448"><a name="zh-cn_topic_0000001549188224_zh-cn_topic_0000001526111046_zh-cn_topic_0000001575944081_p10223674448"></a><a name="zh-cn_topic_0000001549188224_zh-cn_topic_0000001526111046_zh-cn_topic_0000001575944081_p10223674448"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="15.340000000000002%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001549188224_zh-cn_topic_0000001526111046_zh-cn_topic_0000001575944081_p645511218169"><a name="zh-cn_topic_0000001549188224_zh-cn_topic_0000001526111046_zh-cn_topic_0000001575944081_p645511218169"></a><a name="zh-cn_topic_0000001549188224_zh-cn_topic_0000001526111046_zh-cn_topic_0000001575944081_p645511218169"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="67.44%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001549188224_zh-cn_topic_0000001526111046_zh-cn_topic_0000001575944081_p1922337124411"><a name="zh-cn_topic_0000001549188224_zh-cn_topic_0000001526111046_zh-cn_topic_0000001575944081_p1922337124411"></a><a name="zh-cn_topic_0000001549188224_zh-cn_topic_0000001526111046_zh-cn_topic_0000001575944081_p1922337124411"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001549188224_zh-cn_topic_0000001526111046_zh-cn_topic_0000001575944081_row152234713443"><td class="cellrowborder" valign="top" width="17.22%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001549188224_zh-cn_topic_0000001526111046_zh-cn_topic_0000001575944081_p318615392613"><a name="zh-cn_topic_0000001549188224_zh-cn_topic_0000001526111046_zh-cn_topic_0000001575944081_p318615392613"></a><a name="zh-cn_topic_0000001549188224_zh-cn_topic_0000001526111046_zh-cn_topic_0000001575944081_p318615392613"></a>name</p>
</td>
<td class="cellrowborder" valign="top" width="15.340000000000002%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001549188224_zh-cn_topic_0000001526111046_zh-cn_topic_0000001575944081_p320343694214"><a name="zh-cn_topic_0000001549188224_zh-cn_topic_0000001526111046_zh-cn_topic_0000001575944081_p320343694214"></a><a name="zh-cn_topic_0000001549188224_zh-cn_topic_0000001526111046_zh-cn_topic_0000001575944081_p320343694214"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.44%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001549188224_zh-cn_topic_0000001526111046_p096733515614"><a name="zh-cn_topic_0000001549188224_zh-cn_topic_0000001526111046_p096733515614"></a><a name="zh-cn_topic_0000001549188224_zh-cn_topic_0000001526111046_p096733515614"></a>算子输入名称。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0000001549188224_zh-cn_topic_0000001526111046_zh-cn_topic_0000001525424352_section25791320141317"></a>

算子参数定义，OpParamDef实例，具体请参考[OpParamDef](OpParamDef.md)。

## 约束说明<a name="zh-cn_topic_0000001549188224_zh-cn_topic_0000001526111046_zh-cn_topic_0000001525424352_section19165124931511"></a>

参数注册的顺序需要和算子kernel入口函数一致。

