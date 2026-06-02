# SetPadValue\(ISASI\)<a name="ZH-CN_TOPIC_0000002523344770"></a>

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

设置DataCopyPad需要填充的数值。支持的通路如下：

-   GM-\>VECIN/GM-\>VECOUT

## 函数原型<a name="section620mcpsimp"></a>

```
template <typename T, TPosition pos = TPosition::MAX>
__aicore__ inline void SetPadValue(T paddingValue)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table19493819151620"></a>
<table><thead align="left"><tr id="row149311195164"><th class="cellrowborder" valign="top" width="16.49%" id="mcps1.2.4.1.1"><p id="p249381971616"><a name="p249381971616"></a><a name="p249381971616"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="11.93%" id="mcps1.2.4.1.2"><p id="p154931719111611"><a name="p154931719111611"></a><a name="p154931719111611"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="71.58%" id="mcps1.2.4.1.3"><p id="p1049301914167"><a name="p1049301914167"></a><a name="p1049301914167"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row17358528141618"><td class="cellrowborder" valign="top" width="16.49%" headers="mcps1.2.4.1.1 "><p id="p183599282163"><a name="p183599282163"></a><a name="p183599282163"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="11.93%" headers="mcps1.2.4.1.2 "><p id="p113591028121619"><a name="p113591028121619"></a><a name="p113591028121619"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.58%" headers="mcps1.2.4.1.3 "><p id="p1035918287164"><a name="p1035918287164"></a><a name="p1035918287164"></a>填充值的数据类型，与DataCopyPad接口搬运的数据类型一致。</p>
</td>
</tr>
<tr id="row9493141971612"><td class="cellrowborder" valign="top" width="16.49%" headers="mcps1.2.4.1.1 "><p id="p8493919201610"><a name="p8493919201610"></a><a name="p8493919201610"></a>pos</p>
</td>
<td class="cellrowborder" valign="top" width="11.93%" headers="mcps1.2.4.1.2 "><p id="p19493419161614"><a name="p19493419161614"></a><a name="p19493419161614"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.58%" headers="mcps1.2.4.1.3 "><p id="p16640155952513"><a name="p16640155952513"></a><a name="p16640155952513"></a>用于指定DataCopyPad接口搬运过程中从GM搬运数据到哪一个目的地址，目的地址通过逻辑位置来表达。默认值为TPosition::MAX，等效于TPosition::VECIN或TPosition::VECOUT。</p>
<p id="p1214182122518"><a name="p1214182122518"></a><a name="p1214182122518"></a>支持的取值为：</p>
<a name="ul535119418299"></a><a name="ul535119418299"></a><ul id="ul535119418299"><li>TPosition::VECIN、TPosition::VECOUT、TPosition::MAX</li></ul>
</td>
</tr>
</tbody>
</table>

**表 2**  参数说明

<a name="table1055216132132"></a>
<table><thead align="left"><tr id="row105531513121315"><th class="cellrowborder" valign="top" width="16.49%" id="mcps1.2.4.1.1"><p id="p5553171319138"><a name="p5553171319138"></a><a name="p5553171319138"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="11.93%" id="mcps1.2.4.1.2"><p id="p5553151313131"><a name="p5553151313131"></a><a name="p5553151313131"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="71.58%" id="mcps1.2.4.1.3"><p id="p655316136139"><a name="p655316136139"></a><a name="p655316136139"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row5553201314135"><td class="cellrowborder" valign="top" width="16.49%" headers="mcps1.2.4.1.1 "><p id="p8553813111314"><a name="p8553813111314"></a><a name="p8553813111314"></a>paddingValue</p>
</td>
<td class="cellrowborder" valign="top" width="11.93%" headers="mcps1.2.4.1.2 "><p id="p755318134134"><a name="p755318134134"></a><a name="p755318134134"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.58%" headers="mcps1.2.4.1.3 "><p id="p65530137137"><a name="p65530137137"></a><a name="p65530137137"></a>DataCopyPad接口填充的数值，数据与DataCopyPad接口搬运的数据类型一致。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

无

## 调用示例<a name="section1227835243314"></a>

```
uint32_t m_n1 = 32;
uint32_t m_n2 = 31;
uint32_t m_n2Align = n2 % 32 == 0 ? n2 : (n2 / 32 + 1) * 32;

AscendC::LocalTensor<float> srcLocal = m_queInSrc.AllocTensor<float>();
AscendC::DataCopyExtParams dataCopyExtParams;
AscendC::DataCopyPadExtParams<float> padParams;

dataCopyExtParams.blockCount = m_n1;
dataCopyExtParams.blockLen = m_n2 * sizeof(float);
dataCopyExtParams.srcStride = 0;
dataCopyExtParams.dstStride = 0;

padParams.isPad = false;
padParams.leftPadding = 0;
padParams.rightPadding = 1;

AscendC::SetPadValue((float)37);
AscendC::DataCopyPad(srcLocal, m_srcGlobal, dataCopyExtParams, padParams);
```

```
输入数据（srcGm, shape = [32, 31]）：[[1, 1, 1, ..., 1], [1, 1, 1, ..., 1], ... , [1, 1, 1, ..., 1]]
输出数据（dstGm, shape = [32, 32]）：[[1, 1, 1, ..., 1, 37], [1, 1, 1, ..., 1, 37], ... , [1, 1, 1, ..., 1, 37]]
```

```
// 对于不支持使用立即数进行赋值和初始化的数据类型，如下是一个输入类型bfloat16_t的示例：
AscendC::SetPadValue(m_srcGlobal.GetValue(0));
AscendC::DataCopyPad(srcLocal, m_srcGlobal, dataCopyExtParams, padParams);

输入数据（srcGm, shape = [32, 31]）：[[1, 2, 3, ..., 31], [1, 2, 3, ..., 31], ... , [1, 2, 3, ..., 31]]
输出数据（dstGm, shape = [32, 32]）：[[1, 2, 3, ..., 31, 1], [1, 2, 3, ..., 31, 1], ... , [1, 2, 3, ..., 31, 1]]
```

