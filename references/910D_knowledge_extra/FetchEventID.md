# FetchEventID<a name="ZH-CN_TOPIC_0000002523304462"></a>

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

根据HardEvent（硬件类型的同步事件）获取相应可用的TEventID，此接口不会申请TEventID，仅提供可用的TEventID。

## 函数原型<a name="section620mcpsimp"></a>

```
template <HardEvent evt>
__aicore__ inline TEventID FetchEventID()
__aicore__ inline TEventID FetchEventID(HardEvent evt)
```

## 参数说明<a name="section622mcpsimp"></a>

<a name="table193329316393"></a>
<table><thead align="left"><tr id="row123331131153919"><th class="cellrowborder" valign="top" width="12.36%" id="mcps1.1.4.1.1"><p id="p8333133153913"><a name="p8333133153913"></a><a name="p8333133153913"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="12.379999999999999%" id="mcps1.1.4.1.2"><p id="p518118718459"><a name="p518118718459"></a><a name="p518118718459"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="75.26%" id="mcps1.1.4.1.3"><p id="p833353113393"><a name="p833353113393"></a><a name="p833353113393"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row11660173845017"><td class="cellrowborder" valign="top" width="12.36%" headers="mcps1.1.4.1.1 "><p id="p143161011192917"><a name="p143161011192917"></a><a name="p143161011192917"></a>evt</p>
</td>
<td class="cellrowborder" valign="top" width="12.379999999999999%" headers="mcps1.1.4.1.2 "><p id="p9331531865"><a name="p9331531865"></a><a name="p9331531865"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.26%" headers="mcps1.1.4.1.3 "><p id="p143308311967"><a name="p143308311967"></a><a name="p143308311967"></a>HardEvent类型，硬件同步类型。</p>
<p id="p15571184910124"><a name="p15571184910124"></a><a name="p15571184910124"></a>该类型的具体说明请参考<a href="SetFlag-WaitFlag(ISASI).md">SetFlag/WaitFlag(ISASI)</a>中同步类型的说明。</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section633mcpsimp"></a>

相比于[AllocEventID](AllocEventID.md)，FetchEventID适用于临时使用ID的场景，获取ID后，不会对ID进行占用。在一些复杂的使用场景下，需要开发者自行保证使用正确。比如相同流水连续调用SetFlag/WaitFlag，如果两次传入的ID都是使用FetchEventID获取的，因为两者ID相同会出现程序卡死等未定义行为，这时推荐用户使用AllocEventID。

## 返回值说明<a name="section640mcpsimp"></a>

TEventID

## 调用示例<a name="section6191129670"></a>

```
AscendC::TEventID eventIdVToS = GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_S); //需要插scalar等vector的同步，申请对应的HardEvent的ID
AscendC::SetFlag<AscendC::HardEvent::V_S>(eventIdVToS);
AscendC::WaitFlag<AscendC::HardEvent::V_S>(eventIdVToS);
```

