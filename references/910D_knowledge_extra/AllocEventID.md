# AllocEventID<a name="ZH-CN_TOPIC_0000002554344221"></a>

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

用于申请HardEvent（硬件类型同步事件）的TEventID，必须与[ReleaseEventID](ReleaseEventID.md)搭配使用，调用该接口后，会占用申请的TEventID，直至调用ReleaseEventID释放。

## 函数原型<a name="section620mcpsimp"></a>

```
template <HardEvent evt>
__aicore__ inline TEventID AllocEventID()
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table193329316393"></a>
<table><thead align="left"><tr id="row123331131153919"><th class="cellrowborder" valign="top" width="16.29%" id="mcps1.2.3.1.1"><p id="p8333133153913"><a name="p8333133153913"></a><a name="p8333133153913"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="83.71%" id="mcps1.2.3.1.2"><p id="p833353113393"><a name="p833353113393"></a><a name="p833353113393"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row11660173845017"><td class="cellrowborder" valign="top" width="16.29%" headers="mcps1.2.3.1.1 "><p id="p4571114971214"><a name="p4571114971214"></a><a name="p4571114971214"></a>evt</p>
</td>
<td class="cellrowborder" valign="top" width="83.71%" headers="mcps1.2.3.1.2 "><p id="p15571184910124"><a name="p15571184910124"></a><a name="p15571184910124"></a>HardEvent硬件同步类型。该类型的具体说明请参考<a href="SetFlag-WaitFlag(ISASI).md">SetFlag/WaitFlag(ISASI)</a>中同步类型的说明。</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section633mcpsimp"></a>

TEventID有数量限制，使用结束后应该立刻调用ReleaseEventID释放，防止TEventID耗尽。

## 返回值说明<a name="section640mcpsimp"></a>

TEventID

## 调用示例<a name="section642mcpsimp"></a>

```
AscendC::TEventID eventID = GetTPipePtr()->AllocEventID<AscendC::HardEvent::V_S>(); //需要插入scalar等vector的同步，申请对应的HardEvent的ID
AscendC::SetFlag<AscendC::HardEvent::V_S>(eventID);
......
......
......
AscendC::WaitFlag<AscendC::HardEvent::V_S>(eventID);
GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::V_S>(eventID); //释放scalar等vector的同步HardEvent的ID
......
```

