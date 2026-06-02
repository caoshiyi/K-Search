# MetricsProfStart<a name="ZH-CN_TOPIC_0000002554343883"></a>

## 产品支持情况<a name="section1550532418810"></a>

<a name="table38301303189"></a>
<table><thead align="left"><tr id="row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p1883113061818"><a name="p1883113061818"></a><a name="p1883113061818"></a><span id="ph20833205312295"><a name="ph20833205312295"></a><a name="ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p783113012187"><a name="p783113012187"></a><a name="p783113012187"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row1272474920205"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p12300735171314"><a name="p12300735171314"></a><a name="p12300735171314"></a><span id="ph730011352138"><a name="ph730011352138"></a><a name="ph730011352138"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p37256491200"><a name="p37256491200"></a><a name="p37256491200"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="zh-cn_topic_0000001963799138_zh-cn_topic_0000001997078721_section259105813316"></a>

用于设置性能数据采集信号启动，和MetricsProfStop配合使用。使用msProf工具进行算子上板调优时，可在kernel侧代码段前后分别调用MetricsProfStart和MetricsProfStop来指定需要调优的代码段范围。

## 函数原型<a name="zh-cn_topic_0000001963799138_zh-cn_topic_0000001997078721_section2067518173415"></a>

```
__aicore__ inline void MetricsProfStart()
```

## 参数说明<a name="zh-cn_topic_0000001963799138_zh-cn_topic_0000001997078721_section158061867342"></a>

无

## 返回值说明<a name="zh-cn_topic_0000001963799138_zh-cn_topic_0000001997078721_section640mcpsimp"></a>

无

## 约束说明<a name="zh-cn_topic_0000001963799138_zh-cn_topic_0000001997078721_section43265506459"></a>

无

## 调用示例<a name="zh-cn_topic_0000001963799138_zh-cn_topic_0000001997078721_section82241477610"></a>

```
MetricsProfStart();
```

