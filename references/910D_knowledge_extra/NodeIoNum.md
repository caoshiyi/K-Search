# NodeIoNum<a name="ZH-CN_TOPIC_0000002523303740"></a>

## 功能说明<a name="zh-cn_topic_0000001820650060_zh-cn_topic_0000001389787297_section36583473819"></a>

声明算子定义的输入与输出个数

## 函数原型<a name="zh-cn_topic_0000001820650060_zh-cn_topic_0000001389787297_section13230182415108"></a>

```
ContextBuilder &NodeIoNum(size_t inputNum, size_t outputNum)
```

## 参数说明<a name="zh-cn_topic_0000001820650060_zh-cn_topic_0000001389787297_section75395119104"></a>

<a name="zh-cn_topic_0000001820650060_zh-cn_topic_0000001389787297_table111938719446"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001820650060_zh-cn_topic_0000001389787297_row6223476444"><th class="cellrowborder" valign="top" width="17.22%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001820650060_zh-cn_topic_0000001389787297_p10223674448"><a name="zh-cn_topic_0000001820650060_zh-cn_topic_0000001389787297_p10223674448"></a><a name="zh-cn_topic_0000001820650060_zh-cn_topic_0000001389787297_p10223674448"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="15.340000000000002%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001820650060_zh-cn_topic_0000001389787297_p645511218169"><a name="zh-cn_topic_0000001820650060_zh-cn_topic_0000001389787297_p645511218169"></a><a name="zh-cn_topic_0000001820650060_zh-cn_topic_0000001389787297_p645511218169"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="67.44%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001820650060_zh-cn_topic_0000001389787297_p1922337124411"><a name="zh-cn_topic_0000001820650060_zh-cn_topic_0000001389787297_p1922337124411"></a><a name="zh-cn_topic_0000001820650060_zh-cn_topic_0000001389787297_p1922337124411"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001820650060_zh-cn_topic_0000001389787297_row152234713443"><td class="cellrowborder" valign="top" width="17.22%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001820650060_p759124015322"><a name="zh-cn_topic_0000001820650060_p759124015322"></a><a name="zh-cn_topic_0000001820650060_p759124015322"></a>inputNum</p>
</td>
<td class="cellrowborder" valign="top" width="15.340000000000002%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001820650060_zh-cn_topic_0000001389787297_p15663137127"><a name="zh-cn_topic_0000001820650060_zh-cn_topic_0000001389787297_p15663137127"></a><a name="zh-cn_topic_0000001820650060_zh-cn_topic_0000001389787297_p15663137127"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.44%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001820650060_zh-cn_topic_0000001389787297_p2684123934216"><a name="zh-cn_topic_0000001820650060_zh-cn_topic_0000001389787297_p2684123934216"></a><a name="zh-cn_topic_0000001820650060_zh-cn_topic_0000001389787297_p2684123934216"></a>算子IR原型定义中的输入个数</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001820650060_row10125205610321"><td class="cellrowborder" valign="top" width="17.22%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001820650060_p10125145633215"><a name="zh-cn_topic_0000001820650060_p10125145633215"></a><a name="zh-cn_topic_0000001820650060_p10125145633215"></a>outputNum</p>
</td>
<td class="cellrowborder" valign="top" width="15.340000000000002%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001820650060_p111251456123217"><a name="zh-cn_topic_0000001820650060_p111251456123217"></a><a name="zh-cn_topic_0000001820650060_p111251456123217"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.44%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001820650060_p201253567324"><a name="zh-cn_topic_0000001820650060_p201253567324"></a><a name="zh-cn_topic_0000001820650060_p201253567324"></a>算子IR原型定义中的输出个数</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0000001820650060_zh-cn_topic_0000001389787297_section25791320141317"></a>

当前ContextBuilder的对象

## 约束说明<a name="zh-cn_topic_0000001820650060_zh-cn_topic_0000001389787297_section19165124931511"></a>

必须配合[IrInstanceNum](IrInstanceNum.md)一同使用。

## 调用示例<a name="zh-cn_topic_0000001820650060_zh-cn_topic_0000001389787297_section320753512363"></a>

```
auto builder = ContextBuilder().NodeIoNum(5, 3); // 该算子有5个输入，3个输出
```

