# Arrive<a name="ZH-CN_TOPIC_0000002523343550"></a>

## 产品支持情况<a name="section1550532418810"></a>

<a name="zh-cn_topic_0000002554343969_table38301303189"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000002554343969_row20831180131817"><th class="cellrowborder" valign="top" width="57.95%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0000002554343969_p1883113061818"><a name="zh-cn_topic_0000002554343969_p1883113061818"></a><a name="zh-cn_topic_0000002554343969_p1883113061818"></a><span id="zh-cn_topic_0000002554343969_ph20833205312295"><a name="zh-cn_topic_0000002554343969_ph20833205312295"></a><a name="zh-cn_topic_0000002554343969_ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42.05%" id="mcps1.1.3.1.2"><p id="zh-cn_topic_0000002554343969_p783113012187"><a name="zh-cn_topic_0000002554343969_p783113012187"></a><a name="zh-cn_topic_0000002554343969_p783113012187"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000002554343969_row1272474920205"><td class="cellrowborder" valign="top" width="57.95%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000002554343969_p12300735171314"><a name="zh-cn_topic_0000002554343969_p12300735171314"></a><a name="zh-cn_topic_0000002554343969_p12300735171314"></a><span id="zh-cn_topic_0000002554343969_ph730011352138"><a name="zh-cn_topic_0000002554343969_ph730011352138"></a><a name="zh-cn_topic_0000002554343969_ph730011352138"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42.05%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000002554343969_p37256491200"><a name="zh-cn_topic_0000002554343969_p37256491200"></a><a name="zh-cn_topic_0000002554343969_p37256491200"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="zh-cn_topic_0000001526206862_section212607105720"></a>

通知其他等待的AIV，本AIV已经完成其依赖的任务。

## 函数原型<a name="section765814724715"></a>

```
__aicore__ inline void Arrive(uint32_t arriveIndex)
```

## 参数说明<a name="zh-cn_topic_0000001526206862_section129451113125413"></a>

**表 1**  接口参数说明

<a name="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_table111938719446"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_row6223476444"><th class="cellrowborder" valign="top" width="17.22%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p10223674448"><a name="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p10223674448"></a><a name="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p10223674448"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="15.340000000000002%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p645511218169"><a name="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p645511218169"></a><a name="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p645511218169"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="67.44%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p1922337124411"><a name="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p1922337124411"></a><a name="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p1922337124411"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_row152234713443"><td class="cellrowborder" valign="top" width="17.22%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p2340183613156"><a name="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p2340183613156"></a><a name="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p2340183613156"></a>arriveIndex</p>
</td>
<td class="cellrowborder" valign="top" width="15.340000000000002%" headers="mcps1.2.4.1.2 "><p id="p19741912147"><a name="p19741912147"></a><a name="p19741912147"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.44%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p2684123934216"><a name="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p2684123934216"></a><a name="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p2684123934216"></a>该AIV在Arrive组的序号。范围为[0, arriveSize - 1]。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无。

## 约束说明<a name="zh-cn_topic_0000001526206862_section65498832"></a>

该接口支持在循环中使用，但是受限于多核间通信效率要求，循环最大次数不超过1,048,575次。

## 调用示例<a name="zh-cn_topic_0000001526206862_section97001499599"></a>

完整样例参考：[group\_barrier样例](https://gitcode.com/cann/asc-devkit/tree/master/examples/02_features/14_cube_group_management/group_barrier)。

```
if (id >= 0 && id < ARRIVE_NUM) {
  //各种Vector计算逻辑，用户自行实现
  barA.Arrive(id);  // Arrive组中有2个AIV，分别为Block0、1，表示它们已完成任务。
}
```

