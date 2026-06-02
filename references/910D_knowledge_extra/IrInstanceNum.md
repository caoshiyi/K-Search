# IrInstanceNum<a name="ZH-CN_TOPIC_0000002523343554"></a>

## 功能说明<a name="zh-cn_topic_0000001867289933_zh-cn_topic_0000001389787297_section36583473819"></a>

基于算子的IR定义，声明实例化时每种输入的实际个数。

## 函数原型<a name="zh-cn_topic_0000001867289933_zh-cn_topic_0000001389787297_section13230182415108"></a>

```
ContextBuilder &IrInstanceNum(std::vector<uint32_t> instanceNum)
```

## 参数说明<a name="zh-cn_topic_0000001867289933_zh-cn_topic_0000001389787297_section75395119104"></a>

<a name="zh-cn_topic_0000001867289933_zh-cn_topic_0000001389787297_table111938719446"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001867289933_zh-cn_topic_0000001389787297_row6223476444"><th class="cellrowborder" valign="top" width="17.22%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001867289933_zh-cn_topic_0000001389787297_p10223674448"><a name="zh-cn_topic_0000001867289933_zh-cn_topic_0000001389787297_p10223674448"></a><a name="zh-cn_topic_0000001867289933_zh-cn_topic_0000001389787297_p10223674448"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="15.340000000000002%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001867289933_zh-cn_topic_0000001389787297_p645511218169"><a name="zh-cn_topic_0000001867289933_zh-cn_topic_0000001389787297_p645511218169"></a><a name="zh-cn_topic_0000001867289933_zh-cn_topic_0000001389787297_p645511218169"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="67.44%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001867289933_zh-cn_topic_0000001389787297_p1922337124411"><a name="zh-cn_topic_0000001867289933_zh-cn_topic_0000001389787297_p1922337124411"></a><a name="zh-cn_topic_0000001867289933_zh-cn_topic_0000001389787297_p1922337124411"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001867289933_zh-cn_topic_0000001389787297_row152234713443"><td class="cellrowborder" valign="top" width="17.22%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001867289933_zh-cn_topic_0000001389787297_p8563195616313"><a name="zh-cn_topic_0000001867289933_zh-cn_topic_0000001389787297_p8563195616313"></a><a name="zh-cn_topic_0000001867289933_zh-cn_topic_0000001389787297_p8563195616313"></a>instanceNum</p>
</td>
<td class="cellrowborder" valign="top" width="15.340000000000002%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001867289933_zh-cn_topic_0000001389787297_p15663137127"><a name="zh-cn_topic_0000001867289933_zh-cn_topic_0000001389787297_p15663137127"></a><a name="zh-cn_topic_0000001867289933_zh-cn_topic_0000001389787297_p15663137127"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.44%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001867289933_zh-cn_topic_0000001389787297_p2684123934216"><a name="zh-cn_topic_0000001867289933_zh-cn_topic_0000001389787297_p2684123934216"></a><a name="zh-cn_topic_0000001867289933_zh-cn_topic_0000001389787297_p2684123934216"></a>基于算子IR原型定义，按照输入的index顺序声明实例化个数。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0000001867289933_zh-cn_topic_0000001389787297_section25791320141317"></a>

当前ContextBuilder的对象

## 约束说明<a name="zh-cn_topic_0000001867289933_zh-cn_topic_0000001389787297_section19165124931511"></a>

必须配合NodeIoNum一同使用。

## 调用示例<a name="zh-cn_topic_0000001867289933_zh-cn_topic_0000001389787297_section320753512363"></a>

```
context_ascendc::ContextBuilder builder;
(void)builder.NodeIoNum(5,3)   // 算子有5种输入，3种输出
             .IrInstanceNum({1, 1, 3, 1, 1});  // 算子实例化时，index为2的动态类型输入tensor有3个实例
```

