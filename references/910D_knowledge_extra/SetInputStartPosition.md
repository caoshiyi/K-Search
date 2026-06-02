# SetInputStartPosition<a name="ZH-CN_TOPIC_0000002554424137"></a>

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

设置单核上特征矩阵Input载入数据的起始位置。

## 函数原型<a name="section56832818711"></a>

```
__aicore__ inline void SetInputStartPosition(int64_t diStartPos, int64_t mStartPos)
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
<tbody><tr id="row16741411154412"><td class="cellrowborder" valign="top" width="18.6%" headers="mcps1.1.4.1.1 "><p id="p86741211194410"><a name="p86741211194410"></a><a name="p86741211194410"></a>diStartPos</p>
</td>
<td class="cellrowborder" valign="top" width="17.5%" headers="mcps1.1.4.1.2 "><p id="p1967417117449"><a name="p1967417117449"></a><a name="p1967417117449"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="63.9%" headers="mcps1.1.4.1.3 "><p id="p1867491116446"><a name="p1867491116446"></a><a name="p1867491116446"></a>单核上Din方向起始位置。</p>
</td>
</tr>
<tr id="row1674161115447"><td class="cellrowborder" valign="top" width="18.6%" headers="mcps1.1.4.1.1 "><p id="p1867461114414"><a name="p1867461114414"></a><a name="p1867461114414"></a>mStartPos</p>
</td>
<td class="cellrowborder" valign="top" width="17.5%" headers="mcps1.1.4.1.2 "><p id="p11674181164411"><a name="p11674181164411"></a><a name="p11674181164411"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="63.9%" headers="mcps1.1.4.1.3 "><p id="p2674711174412"><a name="p2674711174412"></a><a name="p2674711174412"></a>单核上M方向起始位置。</p>
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
conv3dApi.SetInputStartPosition(diIdxStart, mIdxStart);
```

