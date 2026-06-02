# SetFixPipeConfig<a name="ZH-CN_TOPIC_0000002554343717"></a>

## 产品支持情况<a name="section1550532418810"></a>

<a name="table38301303189"></a>
<table><thead align="left"><tr id="row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p1883113061818"><a name="p1883113061818"></a><a name="p1883113061818"></a><span id="ph20833205312295"><a name="ph20833205312295"></a><a name="ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p783113012187"><a name="p783113012187"></a><a name="p783113012187"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row1272474920205"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p17301775812"><a name="p17301775812"></a><a name="p17301775812"></a><span id="ph2272194216543"><a name="ph2272194216543"></a><a name="ph2272194216543"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p37256491200"><a name="p37256491200"></a><a name="p37256491200"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section618mcpsimp"></a>

[DataCopy](随路量化激活搬运.md)（CO1-\>GM、CO1-\>A1）过程中进行随路量化时，通过调用该接口设置量化流程中tensor量化参数。

## 函数原型<a name="section620mcpsimp"></a>

```
template <typename T>
__aicore__ inline void SetFixPipeConfig(const LocalTensor<T>& reluPre, const LocalTensor<T>& quantPre, bool isUnitFlag = false)
template <typename T, bool setRelu = false>
__aicore__ inline void SetFixPipeConfig(const LocalTensor<T>& preData, bool isUnitFlag = false)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table4835205712588"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001429830437_row118356578583"><th class="cellrowborder" valign="top" width="13.44%" id="mcps1.2.3.1.1"><p id="zh-cn_topic_0000001429830437_p48354572582"><a name="zh-cn_topic_0000001429830437_p48354572582"></a><a name="zh-cn_topic_0000001429830437_p48354572582"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="86.56%" id="mcps1.2.3.1.2"><p id="zh-cn_topic_0000001429830437_p583535795817"><a name="zh-cn_topic_0000001429830437_p583535795817"></a><a name="zh-cn_topic_0000001429830437_p583535795817"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001429830437_row1835857145817"><td class="cellrowborder" valign="top" width="13.44%" headers="mcps1.2.3.1.1 "><p id="p511145143017"><a name="p511145143017"></a><a name="p511145143017"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="86.56%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001429830437_p168351657155818"><a name="zh-cn_topic_0000001429830437_p168351657155818"></a><a name="zh-cn_topic_0000001429830437_p168351657155818"></a>操作数的数据类型。</p>
</td>
</tr>
<tr id="row1648615377"><td class="cellrowborder" valign="top" width="13.44%" headers="mcps1.2.3.1.1 "><p id="p319714448507"><a name="p319714448507"></a><a name="p319714448507"></a>setRelu</p>
</td>
<td class="cellrowborder" valign="top" width="86.56%" headers="mcps1.2.3.1.2 "><p id="p15197204412501"><a name="p15197204412501"></a><a name="p15197204412501"></a>针对设置一个tensor的情况，当setRelu为true时，设置reluPre；反之设置quantPre。setRelu当前仅支持设置为false。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  参数说明

<a name="table8955841508"></a>
<table><thead align="left"><tr id="row15956194105014"><th class="cellrowborder" valign="top" width="13.661366136613662%" id="mcps1.2.4.1.1"><p id="p7956144195014"><a name="p7956144195014"></a><a name="p7956144195014"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="10.35103510351035%" id="mcps1.2.4.1.2"><p id="p1295624145013"><a name="p1295624145013"></a><a name="p1295624145013"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="75.98759875987598%" id="mcps1.2.4.1.3"><p id="p16956144145011"><a name="p16956144145011"></a><a name="p16956144145011"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row5956546509"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p19287714181617"><a name="p19287714181617"></a><a name="p19287714181617"></a>reluPre</p>
</td>
<td class="cellrowborder" valign="top" width="10.35103510351035%" headers="mcps1.2.4.1.2 "><p id="p192871614151615"><a name="p192871614151615"></a><a name="p192871614151615"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.98759875987598%" headers="mcps1.2.4.1.3 "><p id="p16287121461618"><a name="p16287121461618"></a><a name="p16287121461618"></a>源操作数，relu操作时参与计算的tensor，类型为LocalTensor，支持的TPosition为C2PIPE2GM。</p>
<p id="p17556420182814"><a name="p17556420182814"></a><a name="p17556420182814"></a>reluPre为预留参数，暂未启用，为后续的功能扩展做保留，传入一个空LocalTensor即可。</p>
</td>
</tr>
<tr id="row4956154125018"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p142871414131614"><a name="p142871414131614"></a><a name="p142871414131614"></a>quantPre</p>
</td>
<td class="cellrowborder" valign="top" width="10.35103510351035%" headers="mcps1.2.4.1.2 "><p id="p628711148165"><a name="p628711148165"></a><a name="p628711148165"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.98759875987598%" headers="mcps1.2.4.1.3 "><p id="p0287191420164"><a name="p0287191420164"></a><a name="p0287191420164"></a>源操作数，quant tensor，量化操作时参与计算的tensor，类型为LocalTensor，支持的TPosition为C2PIPE2GM。</p>
</td>
</tr>
<tr id="row9486215111718"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p1648712150175"><a name="p1648712150175"></a><a name="p1648712150175"></a>isUnitFlag</p>
</td>
<td class="cellrowborder" valign="top" width="10.35103510351035%" headers="mcps1.2.4.1.2 "><p id="p19487171515178"><a name="p19487171515178"></a><a name="p19487171515178"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.98759875987598%" headers="mcps1.2.4.1.3 "><p id="p3487131516175"><a name="p3487131516175"></a><a name="p3487131516175"></a>UnitFlag配置项，默认值为false。</p>
<a name="ul1049794662411"></a><a name="ul1049794662411"></a><ul id="ul1049794662411"><li>false：关闭UnitFlag配置。</li><li>true：打开UnitFlag配置。</li></ul>
</td>
</tr>
<tr id="row4835151912539"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p14835151916534"><a name="p14835151916534"></a><a name="p14835151916534"></a>preData</p>
</td>
<td class="cellrowborder" valign="top" width="10.35103510351035%" headers="mcps1.2.4.1.2 "><p id="p2835119105317"><a name="p2835119105317"></a><a name="p2835119105317"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.98759875987598%" headers="mcps1.2.4.1.3 "><p id="p1883517194531"><a name="p1883517194531"></a><a name="p1883517194531"></a>支持设置一个Tensor，通过开关控制是relu Tensor还是quant Tensor，支持的TPosition为C2PIPE2GM。当前仅支持传入quant Tensor。</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section633mcpsimp"></a>

quantPre和reluPre必须是Fixpipe Buffer上的Tensor。

## 返回值说明<a name="section640mcpsimp"></a>

无

## 调用示例<a name="section642mcpsimp"></a>

完整示例可参考[完整示例](随路量化激活搬运.md#li178441955134010)。

```
__aicore__inline void SetFPC(const LocalTensor <int32_t>& reluPreTensor, const LocalTensor <int32_t>& quantPreTensor)
{
 
    AscendC::LocalTensor<uint64_t> workA1 = inQueueDeqA1.AllocTensor<uint64_t>();
    uint16_t deqSize = 128; // deq tensor的size
    AscendC::DataCopy(workA1, deqGlobal, deqSize); // deqGlobal为量化系数的gm地址
    AscendC::LocalTensor<uint64_t> deqFB = inQueueDeqFB.AllocTensor<uint64_t>(); // deq tensor在Fix上的地址
    uint16_t fbufBurstLen = deqSize / 128;  // l1->fix, burst_len unit is 128Bytes
    AscendC::DataCopyParams dataCopyParams(1, fbufBurstLen, 0, 0);
    AscendC::DataCopy(deqFB, workA1, dataCopyParams); 通过DataCopy搬入C2PIPE2GM。
    AscendC::SetFixPipeConfig(deqFB); // 设置量化tensor
    AscendC::PipeBarrier<PIPE_FIX>();
}
```

