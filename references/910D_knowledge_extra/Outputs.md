# Outputs<a name="ZH-CN_TOPIC_0000002554423445"></a>

## 功能说明<a name="zh-cn_topic_0000001867289929_zh-cn_topic_0000001389787297_section36583473819"></a>

将void\* 指针的vector设置为KernelContext的output。

## 函数原型<a name="zh-cn_topic_0000001867289929_zh-cn_topic_0000001389787297_section13230182415108"></a>

```
ContextBuilder &Outputs(std::vector<void *> outputs)
```

## 参数说明<a name="zh-cn_topic_0000001867289929_zh-cn_topic_0000001389787297_section75395119104"></a>

<a name="zh-cn_topic_0000001867289929_zh-cn_topic_0000001389787297_table111938719446"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001867289929_zh-cn_topic_0000001389787297_row6223476444"><th class="cellrowborder" valign="top" width="17.22%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001867289929_zh-cn_topic_0000001389787297_p10223674448"><a name="zh-cn_topic_0000001867289929_zh-cn_topic_0000001389787297_p10223674448"></a><a name="zh-cn_topic_0000001867289929_zh-cn_topic_0000001389787297_p10223674448"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="15.340000000000002%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001867289929_zh-cn_topic_0000001389787297_p645511218169"><a name="zh-cn_topic_0000001867289929_zh-cn_topic_0000001389787297_p645511218169"></a><a name="zh-cn_topic_0000001867289929_zh-cn_topic_0000001389787297_p645511218169"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="67.44%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001867289929_zh-cn_topic_0000001389787297_p1922337124411"><a name="zh-cn_topic_0000001867289929_zh-cn_topic_0000001389787297_p1922337124411"></a><a name="zh-cn_topic_0000001867289929_zh-cn_topic_0000001389787297_p1922337124411"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001867289929_zh-cn_topic_0000001389787297_row152234713443"><td class="cellrowborder" valign="top" width="17.22%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001867289929_zh-cn_topic_0000001389787297_p8563195616313"><a name="zh-cn_topic_0000001867289929_zh-cn_topic_0000001389787297_p8563195616313"></a><a name="zh-cn_topic_0000001867289929_zh-cn_topic_0000001389787297_p8563195616313"></a>outputs</p>
</td>
<td class="cellrowborder" valign="top" width="15.340000000000002%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001867289929_zh-cn_topic_0000001389787297_p15663137127"><a name="zh-cn_topic_0000001867289929_zh-cn_topic_0000001389787297_p15663137127"></a><a name="zh-cn_topic_0000001867289929_zh-cn_topic_0000001389787297_p15663137127"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.44%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001867289929_zh-cn_topic_0000001389787297_p2684123934216"><a name="zh-cn_topic_0000001867289929_zh-cn_topic_0000001389787297_p2684123934216"></a><a name="zh-cn_topic_0000001867289929_zh-cn_topic_0000001389787297_p2684123934216"></a>保存输出的void*指针vector</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0000001867289929_zh-cn_topic_0000001389787297_section25791320141317"></a>

当前ContextBuilder的对象

## 约束说明<a name="zh-cn_topic_0000001867289929_zh-cn_topic_0000001389787297_section19165124931511"></a>

无

## 调用示例<a name="zh-cn_topic_0000001867289929_zh-cn_topic_0000001389787297_section320753512363"></a>

```
PlatformInfo platformInfo;
auto contextBuilder = context_ascendc::ContextBuilder().Outputs({nullptr, reinterpret_cast<void *>(&platformInfo)});
```

