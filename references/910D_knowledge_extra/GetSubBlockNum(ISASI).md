# GetSubBlockNum\(ISASI\)<a name="ZH-CN_TOPIC_0000002554344243"></a>

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

[分离模式](基本架构.md#li188191010204418)下，获取一个AI Core上Cube Core（AIC）或者Vector Core（AIV）的数量。

## 函数原型<a name="section620mcpsimp"></a>

```
__aicore__ inline int64_t GetSubBlockNum()
```

## 参数说明<a name="section622mcpsimp"></a>

无

## 返回值说明<a name="section640mcpsimp"></a>

不同Kernel类型下（通过[设置Kernel类型](设置Kernel类型.md)设置），在AIC和AIV上调用该接口的返回值如下：

**表 1**  返回值列表

<a name="table1992694791316"></a>
<table><thead align="left"><tr id="row19927547201318"><th class="cellrowborder" valign="top" width="7.6284743051389725%" id="mcps1.2.8.1.1"><p id="p992712473130"><a name="p992712473130"></a><a name="p992712473130"></a>Kernel类型</p>
</th>
<th class="cellrowborder" valign="top" width="14.587082583483305%" id="mcps1.2.8.1.2"><p id="p11616141219534"><a name="p11616141219534"></a><a name="p11616141219534"></a>KERNEL_TYPE_AIV_ONLY</p>
</th>
<th class="cellrowborder" valign="top" width="14.457108578284345%" id="mcps1.2.8.1.3"><p id="p89482013538"><a name="p89482013538"></a><a name="p89482013538"></a>KERNEL_TYPE_AIC_ONLY</p>
</th>
<th class="cellrowborder" valign="top" width="16.336732653469305%" id="mcps1.2.8.1.4"><p id="p11471830205418"><a name="p11471830205418"></a><a name="p11471830205418"></a>KERNEL_TYPE_MIX_AIC_1_2</p>
</th>
<th class="cellrowborder" valign="top" width="16.166766646670666%" id="mcps1.2.8.1.5"><p id="p927714015412"><a name="p927714015412"></a><a name="p927714015412"></a>KERNEL_TYPE_MIX_AIC_1_1</p>
</th>
<th class="cellrowborder" valign="top" width="15.116976604679063%" id="mcps1.2.8.1.6"><p id="p19486185911920"><a name="p19486185911920"></a><a name="p19486185911920"></a>KERNEL_TYPE_MIX_AIC_1_0</p>
</th>
<th class="cellrowborder" valign="top" width="15.706858628274345%" id="mcps1.2.8.1.7"><p id="p047313121014"><a name="p047313121014"></a><a name="p047313121014"></a>KERNEL_TYPE_MIX_AIV_1_0</p>
</th>
</tr>
</thead>
<tbody><tr id="row992774721313"><td class="cellrowborder" valign="top" width="7.6284743051389725%" headers="mcps1.2.8.1.1 "><p id="p7927194781317"><a name="p7927194781317"></a><a name="p7927194781317"></a>AIV</p>
</td>
<td class="cellrowborder" valign="top" width="14.587082583483305%" headers="mcps1.2.8.1.2 "><p id="p10927134711136"><a name="p10927134711136"></a><a name="p10927134711136"></a>1</p>
</td>
<td class="cellrowborder" valign="top" width="14.457108578284345%" headers="mcps1.2.8.1.3 "><p id="p1592784721319"><a name="p1592784721319"></a><a name="p1592784721319"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="16.336732653469305%" headers="mcps1.2.8.1.4 "><p id="p592716479131"><a name="p592716479131"></a><a name="p592716479131"></a>2</p>
</td>
<td class="cellrowborder" valign="top" width="16.166766646670666%" headers="mcps1.2.8.1.5 "><p id="p7927247151311"><a name="p7927247151311"></a><a name="p7927247151311"></a>1</p>
</td>
<td class="cellrowborder" valign="top" width="15.116976604679063%" headers="mcps1.2.8.1.6 "><p id="p24865591910"><a name="p24865591910"></a><a name="p24865591910"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="15.706858628274345%" headers="mcps1.2.8.1.7 "><p id="p1047151381019"><a name="p1047151381019"></a><a name="p1047151381019"></a>1</p>
</td>
</tr>
<tr id="row9927104711315"><td class="cellrowborder" valign="top" width="7.6284743051389725%" headers="mcps1.2.8.1.1 "><p id="p139274478137"><a name="p139274478137"></a><a name="p139274478137"></a>AIC</p>
</td>
<td class="cellrowborder" valign="top" width="14.587082583483305%" headers="mcps1.2.8.1.2 "><p id="p6927124751318"><a name="p6927124751318"></a><a name="p6927124751318"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="14.457108578284345%" headers="mcps1.2.8.1.3 "><p id="p49271947101310"><a name="p49271947101310"></a><a name="p49271947101310"></a>1</p>
</td>
<td class="cellrowborder" valign="top" width="16.336732653469305%" headers="mcps1.2.8.1.4 "><p id="p139271347111310"><a name="p139271347111310"></a><a name="p139271347111310"></a>1</p>
</td>
<td class="cellrowborder" valign="top" width="16.166766646670666%" headers="mcps1.2.8.1.5 "><p id="p7927204712131"><a name="p7927204712131"></a><a name="p7927204712131"></a>1</p>
</td>
<td class="cellrowborder" valign="top" width="15.116976604679063%" headers="mcps1.2.8.1.6 "><p id="p104861859690"><a name="p104861859690"></a><a name="p104861859690"></a>1</p>
</td>
<td class="cellrowborder" valign="top" width="15.706858628274345%" headers="mcps1.2.8.1.7 "><p id="p14478135101"><a name="p14478135101"></a><a name="p14478135101"></a>-</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section633mcpsimp"></a>

无

## 调用示例<a name="section177231425115410"></a>

```
int64_t subBlockNum = AscendC::GetSubBlockNum();
```

