# SetBias<a name="ZH-CN_TOPIC_0000002523303732"></a>

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

## 功能说明<a name="section2935259242"></a>

设置偏置矩阵Bias。

## 函数原型<a name="section2012220138516"></a>

```
__aicore__ inline void SetBias(const AscendC::GlobalTensor<BiasT>& bias)
```

## 参数说明<a name="section2078510234519"></a>

<a name="table10876123516424"></a>
<table><thead align="left"><tr id="row16894435184216"><th class="cellrowborder" valign="top" width="16.541654165416542%" id="mcps1.1.4.1.1"><p id="p1789433584215"><a name="p1789433584215"></a><a name="p1789433584215"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="19.921992199219922%" id="mcps1.1.4.1.2"><p id="p19894143513427"><a name="p19894143513427"></a><a name="p19894143513427"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="63.53635363536353%" id="mcps1.1.4.1.3"><p id="p5894113516426"><a name="p5894113516426"></a><a name="p5894113516426"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row58944359427"><td class="cellrowborder" valign="top" width="16.541654165416542%" headers="mcps1.1.4.1.1 "><p id="p28941355422"><a name="p28941355422"></a><a name="p28941355422"></a>bias</p>
</td>
<td class="cellrowborder" valign="top" width="19.921992199219922%" headers="mcps1.1.4.1.2 "><p id="p9894535204216"><a name="p9894535204216"></a><a name="p9894535204216"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="63.53635363536353%" headers="mcps1.1.4.1.3 "><p id="p188941435124215"><a name="p188941435124215"></a><a name="p188941435124215"></a>Bias在Global Memory上的地址。<span id="ph15942199192220"><a name="ph15942199192220"></a><a name="ph15942199192220"></a><span id="ph1294215916225"><a name="ph1294215916225"></a><a name="ph1294215916225"></a><span id="ph894279182218"><a name="ph894279182218"></a><a name="ph894279182218"></a>类型为<a href="GlobalTensor.md">GlobalTensor</a>。</span></span></span>偏置矩阵Bias支持的数据类型为：half、bfloat16_t。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section112237312518"></a>

无

## 约束说明<a name="section414020501054"></a>

在卷积计算中，如果涉及偏置矩阵Bias，必须调用此接口；若卷积计算不涉及Bias，则不应调用此接口。

## 调用示例<a name="section09801457159"></a>

```
GlobalTensor<float> biasGm;
biasGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(bias));
if (biasFlag) {
    conv3dApi.SetBias(biasGm);
}
```

