# SetWeight<a name="ZH-CN_TOPIC_0000002554423461"></a>

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

## 功能说明<a name="section62738191536"></a>

设置权重矩阵Weight。

## 函数原型<a name="section222111254318"></a>

```
__aicore__ inline void SetWeight(const AscendC::GlobalTensor<WeightT>& weight)
```

## 参数说明<a name="section15987311038"></a>

<a name="table1559254916418"></a>
<table><thead align="left"><tr id="row5614104924113"><th class="cellrowborder" valign="top" width="16.541654165416542%" id="mcps1.1.4.1.1"><p id="p9614849134116"><a name="p9614849134116"></a><a name="p9614849134116"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="19.921992199219922%" id="mcps1.1.4.1.2"><p id="p2614249134111"><a name="p2614249134111"></a><a name="p2614249134111"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="63.53635363536353%" id="mcps1.1.4.1.3"><p id="p26141949194112"><a name="p26141949194112"></a><a name="p26141949194112"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row136141149104114"><td class="cellrowborder" valign="top" width="16.541654165416542%" headers="mcps1.1.4.1.1 "><p id="p14614154934119"><a name="p14614154934119"></a><a name="p14614154934119"></a>weight</p>
</td>
<td class="cellrowborder" valign="top" width="19.921992199219922%" headers="mcps1.1.4.1.2 "><p id="p1861414493418"><a name="p1861414493418"></a><a name="p1861414493418"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="63.53635363536353%" headers="mcps1.1.4.1.3 "><p id="p1761464914415"><a name="p1761464914415"></a><a name="p1761464914415"></a>Weight在Global Memory上的地址。<span id="ph15942199192220"><a name="ph15942199192220"></a><a name="ph15942199192220"></a><span id="ph1294215916225"><a name="ph1294215916225"></a><a name="ph1294215916225"></a><span id="ph894279182218"><a name="ph894279182218"></a><a name="ph894279182218"></a>类型为<a href="GlobalTensor.md">GlobalTensor</a>。</span></span></span>权重矩阵Weight支持的数据类型为：half、bfloat16_t。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section5825636432"></a>

无

## 约束说明<a name="section123511449834"></a>

无

## 调用示例<a name="section160275412318"></a>

```
GlobalTensor<half> weightGm;
weightGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(weight));
conv3dApi.SetWeight(weightGm);
```

