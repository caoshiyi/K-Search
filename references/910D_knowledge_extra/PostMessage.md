# PostMessage<a name="ZH-CN_TOPIC_0000002523343596"></a>

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

通过AllocMessage接口获取到消息空间地址msg后，构造消息结构体[CubeMsgType](CubeResGroupHandle使用说明.md#table189051237164018)，发送该消息。

## 函数原型<a name="section765814724715"></a>

```
template <PipeMode pipeMode = PipeMode::SCALAR_MODE>
__aicore__ inline uint16_t PostMessage(__gm__ CubeMsgType* msg, CubeMsgType& msgInput)
```

## 参数说明<a name="zh-cn_topic_0000001526206862_section129451113125413"></a>

**表 1**  模板参数说明

<a name="table1436511617117"></a>
<table><thead align="left"><tr id="row4365369115"><th class="cellrowborder" valign="top" width="12.82%" id="mcps1.2.3.1.1"><p id="p236526181117"><a name="p236526181117"></a><a name="p236526181117"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="87.18%" id="mcps1.2.3.1.2"><p id="p236514615117"><a name="p236514615117"></a><a name="p236514615117"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row14365116121111"><td class="cellrowborder" valign="top" width="12.82%" headers="mcps1.2.3.1.1 "><p id="p23653641116"><a name="p23653641116"></a><a name="p23653641116"></a>pipeMode</p>
</td>
<td class="cellrowborder" valign="top" width="87.18%" headers="mcps1.2.3.1.2 "><p id="p1213214123127"><a name="p1213214123127"></a><a name="p1213214123127"></a>用于配置发送消息的执行单元。PipeMode类型，其定义如下：</p>
<a name="screen11132212171216"></a><a name="screen11132212171216"></a><pre class="screen" codetype="Cpp" id="screen11132212171216">enum class PipeMode : uint8_t { 
  SCALAR_MODE = 0, // Scalar执行单元往GM上写消息。
  MTE3_MODE = 1, // 使用MTE3单元往GM上写消息。
  MAX 
}</pre>
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
<tbody><tr id="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_row152234713443"><td class="cellrowborder" valign="top" width="9.09%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p2340183613156"><a name="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p2340183613156"></a><a name="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p2340183613156"></a>msg</p>
</td>
<td class="cellrowborder" valign="top" width="7.93%" headers="mcps1.2.4.1.2 "><p id="p19741912147"><a name="p19741912147"></a><a name="p19741912147"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="82.98%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p2684123934216"><a name="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p2684123934216"></a><a name="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p2684123934216"></a>该CubeResGroupHandle中某个任务的消息空间地址。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001526206862_row1239183183016"><td class="cellrowborder" valign="top" width="9.09%" headers="mcps1.2.4.1.1 "><p id="p16720185325117"><a name="p16720185325117"></a><a name="p16720185325117"></a>msgInput</p>
</td>
<td class="cellrowborder" valign="top" width="7.93%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001526206862_p7239938308"><a name="zh-cn_topic_0000001526206862_p7239938308"></a><a name="zh-cn_topic_0000001526206862_p7239938308"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="82.98%" headers="mcps1.2.4.1.3 "><p id="p8307185573018"><a name="p8307185573018"></a><a name="p8307185573018"></a>需要发送的消息内容。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

当前消息空间与该消息队列队首空间的地址偏移。

## 约束说明<a name="zh-cn_topic_0000001526206862_section65498832"></a>

无

## 调用示例<a name="zh-cn_topic_0000001526206862_section97001499599"></a>

```
handle.AssignQueue(queIdx);  
auto msgPtr = handle.AllocMessage();        // 获取消息空间指针msgPtr
AscendC::CubeGroupMsgHead headA = {AscendC::CubeMsgState::VALID, 0};
AscendC::CubeMsgBody msgA = {headA, 1, 0, 0, false, false, false, false, 0, 0, 0, 0, 0, 0, 0, 0};
auto offset = handle.PostMessage(msgPtr, msgA);           // 在msgPtr指针位置，填充用户自定义的消息结构体，并发送
```

