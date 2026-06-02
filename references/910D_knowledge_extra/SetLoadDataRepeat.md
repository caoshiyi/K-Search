# SetLoadDataRepeat<a name="ZH-CN_TOPIC_0000002523343530"></a>

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

用于设置[Load3Dv2接口](Load3D.md#li83241850104315)的repeat参数。设置repeat参数后，可以通过调用一次Load3Dv2接口完成多个迭代的数据搬运。

## 函数原型<a name="section620mcpsimp"></a>

```
__aicore__ inline void SetLoadDataRepeat(const LoadDataRepeatParam& repeatParams)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  参数说明

<a name="table8955841508"></a>
<table><thead align="left"><tr id="row15956194105014"><th class="cellrowborder" valign="top" width="18.81188118811881%" id="mcps1.2.4.1.1"><p id="p7956144195014"><a name="p7956144195014"></a><a name="p7956144195014"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="13.201320132013203%" id="mcps1.2.4.1.2"><p id="p1295624145013"><a name="p1295624145013"></a><a name="p1295624145013"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="67.98679867986797%" id="mcps1.2.4.1.3"><p id="p16956144145011"><a name="p16956144145011"></a><a name="p16956144145011"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row1075785651510"><td class="cellrowborder" valign="top" width="18.81188118811881%" headers="mcps1.2.4.1.1 "><p id="p1728791441620"><a name="p1728791441620"></a><a name="p1728791441620"></a>repeatParams</p>
</td>
<td class="cellrowborder" valign="top" width="13.201320132013203%" headers="mcps1.2.4.1.2 "><p id="p11287151451610"><a name="p11287151451610"></a><a name="p11287151451610"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.98679867986797%" headers="mcps1.2.4.1.3 "><p id="p17376814155615"><a name="p17376814155615"></a><a name="p17376814155615"></a>设置Load3Dv2接口的repeat参数，类型为LoadDataRepeatParam。</p>
<p id="p595519531047"><a name="p595519531047"></a><a name="p595519531047"></a>具体定义请参考<span id="ph10562197165916"><a name="ph10562197165916"></a><a name="ph10562197165916"></a>${INSTALL_DIR}</span>/include/ascendc/basic_api/interface/kernel_struct_mm.h，<span id="ph14322531015"><a name="ph14322531015"></a><a name="ph14322531015"></a>${INSTALL_DIR}</span>请替换为CANN软件安装后文件存储路径。</p>
<p id="p12287014111614"><a name="p12287014111614"></a><a name="p12287014111614"></a>参数说明请参考<a href="#table15780447181917">表2</a>。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  LoadDataRepeatParam结构体参数说明

<a name="table15780447181917"></a>
<table><thead align="left"><tr id="row0780947111915"><th class="cellrowborder" valign="top" width="15.25%" id="mcps1.2.3.1.1"><p id="p1780124771913"><a name="p1780124771913"></a><a name="p1780124771913"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="84.75%" id="mcps1.2.3.1.2"><p id="p1578014718198"><a name="p1578014718198"></a><a name="p1578014718198"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row13851184516228"><td class="cellrowborder" valign="top" width="15.25%" headers="mcps1.2.3.1.1 "><p id="p164179505229"><a name="p164179505229"></a><a name="p164179505229"></a>repeatTime</p>
</td>
<td class="cellrowborder" valign="top" width="84.75%" headers="mcps1.2.3.1.2 "><p id="p8417135012216"><a name="p8417135012216"></a><a name="p8417135012216"></a>height/width方向上的迭代次数，取值范围：repeatTime ∈[0, 255] 。默认值为1。</p>
</td>
</tr>
<tr id="row10780647151919"><td class="cellrowborder" valign="top" width="15.25%" headers="mcps1.2.3.1.1 "><p id="p6340835122118"><a name="p6340835122118"></a><a name="p6340835122118"></a>repeatStride</p>
</td>
<td class="cellrowborder" valign="top" width="84.75%" headers="mcps1.2.3.1.2 "><p id="p12340173514212"><a name="p12340173514212"></a><a name="p12340173514212"></a>height/width方向上的前一个迭代与后一个迭代起始地址的距离，取值范围：n∈[0, 65535]，默认值为0。</p>
<a name="ul146791315738"></a><a name="ul146791315738"></a><ul id="ul146791315738"><li>repeatMode为0，repeatStride的单位为16个元素。</li><li>repeatMode为1，repeatStride的单位和具体型号有关。下文中的data_type指Load3Dv2中源操作数的数据类型。<p id="p1140374875318"><a name="p1140374875318"></a><a name="p1140374875318"></a><span id="ph1675675114341"><a name="ph1675675114341"></a><a name="ph1675675114341"></a>Ascend 950PR/Ascend 950DT</span>，repeatStride的单位为32/sizeof(data_type)个元素 。</p>
</li></ul>
</td>
</tr>
<tr id="row1078074711194"><td class="cellrowborder" valign="top" width="15.25%" headers="mcps1.2.3.1.1 "><p id="p334033518217"><a name="p334033518217"></a><a name="p334033518217"></a>repeatMode</p>
</td>
<td class="cellrowborder" valign="top" width="84.75%" headers="mcps1.2.3.1.2 "><p id="p1734053517219"><a name="p1734053517219"></a><a name="p1734053517219"></a>控制repeat迭代的方向，取值范围：k∈[0, 1] 。默认值为0。</p>
<p id="p9880145919259"><a name="p9880145919259"></a><a name="p9880145919259"></a>0：迭代沿height方向；</p>
<p id="p432516193267"><a name="p432516193267"></a><a name="p432516193267"></a>1：迭代沿width方向。</p>
</td>
</tr>
</tbody>
</table>

