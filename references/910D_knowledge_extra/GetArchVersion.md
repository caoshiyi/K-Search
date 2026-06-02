# GetArchVersion<a name="ZH-CN_TOPIC_0000002554344267"></a>

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

获取当前AI处理器架构版本号。

## 函数原型<a name="section620mcpsimp"></a>

```
__aicore__ inline void GetArchVersion(uint32_t& coreVersion)
```

## 参数说明<a name="section622mcpsimp"></a>

<a name="table1151916565167"></a>
<table><thead align="left"><tr id="row125391056181610"><th class="cellrowborder" valign="top" width="33.33%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="19.57%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="47.099999999999994%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row153995681612"><td class="cellrowborder" valign="top" width="33.33%" headers="mcps1.1.4.1.1 "><p id="p145399565160"><a name="p145399565160"></a><a name="p145399565160"></a>coreVersion</p>
</td>
<td class="cellrowborder" valign="top" width="19.57%" headers="mcps1.1.4.1.2 "><p id="p1653915567162"><a name="p1653915567162"></a><a name="p1653915567162"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="47.099999999999994%" headers="mcps1.1.4.1.3 "><p id="p12586181110243"><a name="p12586181110243"></a><a name="p12586181110243"></a>AI处理器架构版本</p>
<p id="p7801124616151"><a name="p7801124616151"></a><a name="p7801124616151"></a>数据类型：uint32_t</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

在调用GetArchVersion接口前，需先定义coreVersion ，调用GetArchVersion接口后coreVersion会变成相对应架构版本号的值。

由于硬件约束，在查看转换后的AI处理器架构版本号时需要将其打印成十六进制的数或者自行转换成十六进制的数。

## 调用示例<a name="section177231425115410"></a>

如下样例通过调用GetArchVersion接口获取AI处理器架构版本号。

```
    uint32_t coreVersion = 0;//定义AI处理器版本
    AscendC::GetArchVersion(coreVersion);
    AscendC::PRINTF("core version is %x", coreVersion);//需用%x将其打印成十六进制的数
```

不同型号服务器有不同的架构版本号取值，如下表所示：

<a name="table480419223573"></a>
<table><thead align="left"><tr id="row15804162215719"><th class="cellrowborder" valign="top" width="48.49%" id="mcps1.1.3.1.1"><p id="p1680412205710"><a name="p1680412205710"></a><a name="p1680412205710"></a>架构版本号</p>
</th>
<th class="cellrowborder" valign="top" width="51.51%" id="mcps1.1.3.1.2"><p id="p15804152219577"><a name="p15804152219577"></a><a name="p15804152219577"></a>型号</p>
</th>
</tr>
</thead>
<tbody><tr id="row10206416131819"><td class="cellrowborder" valign="top" width="48.49%" headers="mcps1.1.3.1.1 "><p id="p5880145213281"><a name="p5880145213281"></a><a name="p5880145213281"></a>暂未正式确定，请以最终商发版本说明为准。</p>
</td>
<td class="cellrowborder" valign="top" width="51.51%" headers="mcps1.1.3.1.2 "><p id="p387963321813"><a name="p387963321813"></a><a name="p387963321813"></a><span id="ph158791833111819"><a name="ph158791833111819"></a><a name="ph158791833111819"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
</tr>
</tbody>
</table>

