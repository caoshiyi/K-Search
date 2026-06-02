# 原型注册接口（OP\_ADD）<a name="ZH-CN_TOPIC_0000002523343950"></a>

## 功能说明<a name="zh-cn_topic_0000001600307121_zh-cn_topic_0000001576870453_zh-cn_topic_0000001525424352_section36583473819"></a>

注册算子的原型定义，从而确保算子能够被框架正确识别、编译和执行****。

算子原型主要描述了算子的输入输出、属性等信息以及算子在AI处理器上相关实现信息，并关联[tiling实现](Host侧Tiling实现.md)等函数。算子原型通过自定义的算子类来承载，该算子类继承自[OpDef类](OpDef.md)。完成算子的原型定义等操作后，需要调用[OP\_ADD](原型注册接口（OP_ADD）.md)接口，传入算子类型（自定义算子类的类名），进行算子原型注册。详细内容请参考[算子原型定义](算子原型定义.md)。

## 函数原型<a name="zh-cn_topic_0000001600307121_zh-cn_topic_0000001576870453_zh-cn_topic_0000001525424352_section13230182415108"></a>

```
OP_ADD(opType)
```

## 参数说明<a name="zh-cn_topic_0000001600307121_zh-cn_topic_0000001576870453_zh-cn_topic_0000001525424352_section75395119104"></a>

<a name="zh-cn_topic_0000001600307121_zh-cn_topic_0000001576870453_zh-cn_topic_0000001575944081_table111938719446"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001600307121_zh-cn_topic_0000001576870453_zh-cn_topic_0000001575944081_row6223476444"><th class="cellrowborder" valign="top" width="17.22%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001600307121_zh-cn_topic_0000001576870453_zh-cn_topic_0000001575944081_p10223674448"><a name="zh-cn_topic_0000001600307121_zh-cn_topic_0000001576870453_zh-cn_topic_0000001575944081_p10223674448"></a><a name="zh-cn_topic_0000001600307121_zh-cn_topic_0000001576870453_zh-cn_topic_0000001575944081_p10223674448"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="15.340000000000002%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001600307121_zh-cn_topic_0000001576870453_zh-cn_topic_0000001575944081_p645511218169"><a name="zh-cn_topic_0000001600307121_zh-cn_topic_0000001576870453_zh-cn_topic_0000001575944081_p645511218169"></a><a name="zh-cn_topic_0000001600307121_zh-cn_topic_0000001576870453_zh-cn_topic_0000001575944081_p645511218169"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="67.44%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001600307121_zh-cn_topic_0000001576870453_zh-cn_topic_0000001575944081_p1922337124411"><a name="zh-cn_topic_0000001600307121_zh-cn_topic_0000001576870453_zh-cn_topic_0000001575944081_p1922337124411"></a><a name="zh-cn_topic_0000001600307121_zh-cn_topic_0000001576870453_zh-cn_topic_0000001575944081_p1922337124411"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001600307121_zh-cn_topic_0000001576870453_zh-cn_topic_0000001575944081_row152234713443"><td class="cellrowborder" valign="top" width="17.22%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001600307121_zh-cn_topic_0000001576870453_zh-cn_topic_0000001575944081_p318615392613"><a name="zh-cn_topic_0000001600307121_zh-cn_topic_0000001576870453_zh-cn_topic_0000001575944081_p318615392613"></a><a name="zh-cn_topic_0000001600307121_zh-cn_topic_0000001576870453_zh-cn_topic_0000001575944081_p318615392613"></a>opType</p>
</td>
<td class="cellrowborder" valign="top" width="15.340000000000002%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001600307121_zh-cn_topic_0000001576870453_zh-cn_topic_0000001575944081_p320343694214"><a name="zh-cn_topic_0000001600307121_zh-cn_topic_0000001576870453_zh-cn_topic_0000001575944081_p320343694214"></a><a name="zh-cn_topic_0000001600307121_zh-cn_topic_0000001576870453_zh-cn_topic_0000001575944081_p320343694214"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.44%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001600307121_zh-cn_topic_0000001576870453_p096733515614"><a name="zh-cn_topic_0000001600307121_zh-cn_topic_0000001576870453_p096733515614"></a><a name="zh-cn_topic_0000001600307121_zh-cn_topic_0000001576870453_p096733515614"></a>算子类型名称</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0000001600307121_zh-cn_topic_0000001576870453_zh-cn_topic_0000001525424352_section25791320141317"></a>

无

## 约束说明<a name="zh-cn_topic_0000001600307121_zh-cn_topic_0000001576870453_zh-cn_topic_0000001525424352_section19165124931511"></a>

无

