# AllocMessage<a name="ZH-CN_TOPIC_0000002523343914"></a>

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

AIV从消息队列里申请消息空间，用于存放消息结构体，返回当前申请的消息空间的地址。消息队列的深度固定为4，申请消息空间的顺序为自上而下，然后循环。当消息队列指针指向的消息空间为FREE状态时，AllocMessage返回空间的地址，否则循环等待，直到当前空间的状态为FREE。

**图 1**  AllocMessage示意图<a name="fig153760010192"></a>  
<!-- img2text -->
```text
┌───────────────────────┬─────────┬──────────────┬─────────┬─────────┬─────────┐
│ CubeResGroup          │ Block 0 │   Queue 0    │ Queue 1 │ Queue 2 │ Queue 3 │ Queue 4 │
│ Handle                │         ├──────────────┼─────────┼─────────┼─────────┼─────────┤
│                       │         │   Queue 0    │ Queue 1 │ Queue 2 │ Queue 3 │ Queue 4 │
│                       │         ├──────────────┼─────────┼─────────┼─────────┼─────────┤
│                       │         │   Queue 0    │ Queue 1 │ Queue 2 │ Queue 3 │ Queue 4 │
│                       │         ├──────────────┼─────────┼─────────┼─────────┼─────────┤
│                       │         │   Queue 0    │ Queue 1 │ Queue 2 │ Queue 3 │ Queue 4 │
├───────────────────────┼─────────┼──────────────┼─────────┼─────────┼─────────┼─────────┤
│                       │ Block 1 │ Queue 5(0)   │ Queue 6 │ Queue 7 │ Queue 8 │ Queue 9 │
│                       │         │      │       │         │         │         │         │
│                       │         │      ↓       │         │         │         │         │
│                       │         │ Queue 5(1)   │ Queue 6 │ Queue 7 │ Queue 8 │ Queue 9 │
│                       │         │      │       │         │         │         │         │
│                       │         │      ↓       │         │         │         │         │
│                       │         │ Queue 5(2)   │ Queue 6 │ Queue 7 │ Queue 8 │ Queue 9 │
│                       │         │      │       │         │         │         │         │
│                       │         │      ↓       │         │         │         │         │
│                       │         │ Queue 5(3)   │ Queue 6 │ Queue 7 │ Queue 8 │ Queue 9 │
└───────────────────────┴─────────┴──────────────┴─────────┴─────────┴─────────┴─────────┘
```

## 函数原型<a name="section765814724715"></a>

```
template <PipeMode pipeMode = PipeMode::SCALAR_MODE>             
__aicore__ inline __gm__ CubeMsgType *AllocMessage()
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
<p id="p168827373292"><a name="p168827373292"></a><a name="p168827373292"></a></p>
<p id="p2132171218121"><a name="p2132171218121"></a><a name="p2132171218121"></a>注意，pipeMode为MTE3_MODE时，后续只能使用PostMessage接口发送消息。同时两个接口AllocMessage与PostMessage的模板参数pipeMode需要相同。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

当前申请的消息空间的地址。

## 约束说明<a name="zh-cn_topic_0000001526206862_section65498832"></a>

无

## 调用示例<a name="zh-cn_topic_0000001526206862_section97001499599"></a>

```
auto queIdx = AscendC::GetBlockIdx();
handle.AssignQueue(queIdx);
auto msgPtr = handle.AllocMessage();        // 绑定队列后，从该队列中申请消息空间，消息空间地址为msgPtr。
```

