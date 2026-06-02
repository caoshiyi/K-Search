# Wait<a name="ZH-CN_TOPIC_0000002554343847"></a>

## 产品支持情况<a name="section1550532418810"></a>

<a name="zh-cn_topic_0000002554343969_table38301303189"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000002554343969_row20831180131817"><th class="cellrowborder" valign="top" width="57.95%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0000002554343969_p1883113061818"><a name="zh-cn_topic_0000002554343969_p1883113061818"></a><a name="zh-cn_topic_0000002554343969_p1883113061818"></a><span id="zh-cn_topic_0000002554343969_ph20833205312295"><a name="zh-cn_topic_0000002554343969_ph20833205312295"></a><a name="zh-cn_topic_0000002554343969_ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42.05%" id="mcps1.1.3.1.2"><p id="zh-cn_topic_0000002554343969_p783113012187"><a name="zh-cn_topic_0000002554343969_p783113012187"></a><a name="zh-cn_topic_0000002554343969_p783113012187"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000002554343969_row1272474920205"><td class="cellrowborder" valign="top" width="57.95%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000002554343969_p12300735171314"><a name="zh-cn_topic_0000002554343969_p12300735171314"></a><a name="zh-cn_topic_0000002554343969_p12300735171314"></a><span id="zh-cn_topic_0000002554343969_ph730011352138"><a name="zh-cn_topic_0000002554343969_ph730011352138"></a><a name="zh-cn_topic_0000002554343969_ph730011352138"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42.05%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000002554343969_p37256491200"><a name="zh-cn_topic_0000002554343969_p37256491200"></a><a name="zh-cn_topic_0000002554343969_p37256491200"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="zh-cn_topic_0000001526206862_section212607105720"></a>

在调用PostMessage或PostFakeMessage后，查询该消息是否已被AIC处理完。

## 函数原型<a name="section765814724715"></a>

```
template <bool sync = true>
__aicore__ inline bool Wait(uint16_t offset)
```

## 参数说明<a name="zh-cn_topic_0000001526206862_section129451113125413"></a>

**表 1**  模板参数说明

<a name="table7254433348"></a>
<table><thead align="left"><tr id="row102594383418"><th class="cellrowborder" valign="top" width="9.2%" id="mcps1.2.3.1.1"><p id="p102574316348"><a name="p102574316348"></a><a name="p102574316348"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="90.8%" id="mcps1.2.3.1.2"><p id="p325194343417"><a name="p325194343417"></a><a name="p325194343417"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row525343163416"><td class="cellrowborder" valign="top" width="9.2%" headers="mcps1.2.3.1.1 "><p id="p2251243143411"><a name="p2251243143411"></a><a name="p2251243143411"></a>sync</p>
</td>
<td class="cellrowborder" valign="top" width="90.8%" headers="mcps1.2.3.1.2 "><p id="p10936103116363"><a name="p10936103116363"></a><a name="p10936103116363"></a>查询消息时，程序的运行是否需要等待。参数取值如下：</p>
<a name="ul9461719193513"></a><a name="ul9461719193513"></a><ul id="ul9461719193513"><li>true，必须等到AIC处理完该消息后，程序才可以继续运行。</li><li>false，仅查询AIC是否处理完该消息。</li></ul>
</td>
</tr>
</tbody>
</table>

**表 2**  接口参数说明

<a name="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_table111938719446"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_row6223476444"><th class="cellrowborder" valign="top" width="9.09%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p10223674448"><a name="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p10223674448"></a><a name="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p10223674448"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="7.93%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p645511218169"><a name="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p645511218169"></a><a name="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p645511218169"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="82.98%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p1922337124411"><a name="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p1922337124411"></a><a name="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p1922337124411"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row6413133313219"><td class="cellrowborder" valign="top" width="9.09%" headers="mcps1.2.4.1.1 "><p id="p7413153317217"><a name="p7413153317217"></a><a name="p7413153317217"></a>offset</p>
</td>
<td class="cellrowborder" valign="top" width="7.93%" headers="mcps1.2.4.1.2 "><p id="p741317331213"><a name="p741317331213"></a><a name="p741317331213"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="82.98%" headers="mcps1.2.4.1.3 "><p id="p1660161516133"><a name="p1660161516133"></a><a name="p1660161516133"></a>消息空间地址偏移量，通过PostMessage或者PostFakeMessage的返回值获取。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

-   true：当前消息已被AIC处理完。
-   false：当前消息未被AIC处理完。

## 约束说明<a name="zh-cn_topic_0000001526206862_section65498832"></a>

无

## 调用示例<a name="zh-cn_topic_0000001526206862_section97001499599"></a>

```
auto msgPtr = handle.AllocMessage();        // 在msgPtr指针这个位置，可以发送一个新消息
AscendC::CubeGroupMsgHead headA = {AscendC::CubeMsgState::VALID, 0};
AscendC::CubeMsgBody msgA = {headA, 1, 0, 0, false, false, false, false, 0, 0, 0, 0, 0, 0, 0, 0};
auto offset = handle.PostMessage(msgPtr, msgA);           // 在msgPtr指针位置，填充用户自定义的消息结构体，并发送
bool waitState = handle.template Wait<true>(offset);      // 等待AIC处理完msgA
// 假消息场景
auto msgFakePtr = handle.AllocMessage();
offset = handle.PostFakeMsg(msgFakePtr);
bool waitState = handle.template Wait<true>(offset); // 等待AIC处理完假消息msgFake
```

