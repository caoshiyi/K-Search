# SetInput<a name="ZH-CN_TOPIC_0000002523344354"></a>

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

## 功能说明<a name="section189715191728"></a>

设置特征矩阵Input。

## 函数原型<a name="section15102102712219"></a>

```
__aicore__ inline void SetInput(const AscendC::GlobalTensor<InputT>& input)
```

## 参数说明<a name="section108851126942"></a>

<a name="table752219114412"></a>
<table><thead align="left"><tr id="row75381215415"><th class="cellrowborder" valign="top" width="16.541654165416542%" id="mcps1.1.4.1.1"><p id="p65381118416"><a name="p65381118416"></a><a name="p65381118416"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="17.461746174617463%" id="mcps1.1.4.1.2"><p id="p1653817110410"><a name="p1653817110410"></a><a name="p1653817110410"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="65.99659965996601%" id="mcps1.1.4.1.3"><p id="p1253811154117"><a name="p1253811154117"></a><a name="p1253811154117"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row15538101164114"><td class="cellrowborder" valign="top" width="16.541654165416542%" headers="mcps1.1.4.1.1 "><p id="p753811174113"><a name="p753811174113"></a><a name="p753811174113"></a>input</p>
</td>
<td class="cellrowborder" valign="top" width="17.461746174617463%" headers="mcps1.1.4.1.2 "><p id="p253811144111"><a name="p253811144111"></a><a name="p253811144111"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="65.99659965996601%" headers="mcps1.1.4.1.3 "><p id="p1153819119417"><a name="p1153819119417"></a><a name="p1153819119417"></a>Input在Global Memory上的首地址。<span id="ph15942199192220"><a name="ph15942199192220"></a><a name="ph15942199192220"></a><span id="ph1294215916225"><a name="ph1294215916225"></a><a name="ph1294215916225"></a><span id="ph894279182218"><a name="ph894279182218"></a><a name="ph894279182218"></a>类型为<a href="GlobalTensor.md">GlobalTensor</a>。</span></span></span>特征矩阵Input支持的数据类型为：half、bfloat16_t。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section137241837924"></a>

无

## 约束说明<a name="section020216531924"></a>

无

## 调用示例<a name="section66051011635"></a>

```
GlobalTensor<half> inputGm;
inputGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(input));
conv3dApi.SetInput(inputGm);
```

