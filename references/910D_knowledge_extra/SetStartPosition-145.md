# SetStartPosition<a name="ZH-CN_TOPIC_0000002554423643"></a>

## 产品支持情况<a name="section1586581915393"></a>

<a name="table169596713360"></a>
<table><thead align="left"><tr id="row129590715369"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p17959971362"><a name="p17959971362"></a><a name="p17959971362"></a><span id="ph895914718367"><a name="ph895914718367"></a><a name="ph895914718367"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p89594763612"><a name="p89594763612"></a><a name="p89594763612"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row18959673369"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p1595910763613"><a name="p1595910763613"></a><a name="p1595910763613"></a><span id="ph1595918753613"><a name="ph1595918753613"></a><a name="ph1595918753613"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p1695957133611"><a name="p1695957133611"></a><a name="p1695957133611"></a>x</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section5658171778"></a>

设置单核上GradOutput载入数据的起始位置。

## 函数原型<a name="section56832818711"></a>

```
__aicore__ inline void SetStartPosition(uint32_t hoStartIdx)
```

## 参数说明<a name="section108214161073"></a>

<a name="table156221011124411"></a>
<table><thead align="left"><tr id="row36743117440"><th class="cellrowborder" valign="top" width="18.6%" id="mcps1.1.4.1.1"><p id="p126747111440"><a name="p126747111440"></a><a name="p126747111440"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="17.5%" id="mcps1.1.4.1.2"><p id="p4674111174414"><a name="p4674111174414"></a><a name="p4674111174414"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="63.9%" id="mcps1.1.4.1.3"><p id="p13674131184418"><a name="p13674131184418"></a><a name="p13674131184418"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row16741411154412"><td class="cellrowborder" valign="top" width="18.6%" headers="mcps1.1.4.1.1 "><p id="p2504167270"><a name="p2504167270"></a><a name="p2504167270"></a>hoStartIdx</p>
</td>
<td class="cellrowborder" valign="top" width="17.5%" headers="mcps1.1.4.1.2 "><p id="p1150121612274"><a name="p1150121612274"></a><a name="p1150121612274"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="63.9%" headers="mcps1.1.4.1.3 "><p id="p450116132713"><a name="p450116132713"></a><a name="p450116132713"></a>当前核GradOutput Height方向起始位置。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section37846231973"></a>

无

## 约束说明<a name="section374517371071"></a>

无

## 调用示例<a name="section1994119441271"></a>

```
...
gradWeight_.SetSingleShape(singleShapeM, singleShapeN, singleShapeK);
gradWeight_.SetStartPosition(hoStartIdx); // 设置单核上GradOutput载入的起始位置
...
```

