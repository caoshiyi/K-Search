# PostFakeMsg<a name="ZH-CN_TOPIC_0000002523344630"></a>

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

通过AllocMessage接口获取到消息空间地址后，AIV发送假消息，刷新消息状态msgState为FAKE。

当多个AIV的消息内容一致时，AIC仅需要读取一次位置靠前的第一个消息，通过将消息结构体中自定义的参数skipCnt设置为n，通知AIC后续n条消息无需处理，直接跳过，被跳过的AIV需要使用本接口发送假消息，这被称之为消息合并机制或消息合并场景。

如下图所示，假设Queue1、2、3的第0条消息与Queue0的第0条消息相同，在消息合并场景中，从AIC视角来看，Queue0\(0\)，Queue4\(0\)的消息会被处理，并根据用户自定义的消息内容完成相应的AIC上的计算。Queue1\(0\), Queue2\(0\), Queue3\(0\)由于发了假消息，AIC将不会读取消息内容进行计算，直接释放消息。

**图 1**  PostFakeMessage示意图<a name="fig6289195515216"></a>  
<!-- img2text -->
```text
┌──────────────────────┬────────┬────────────────────┬───────────────────┬───────────────────┬───────────────────┬────────────────────┐
│ CubeResGroup         │        │                    │                   │                   │                   │                    │
│ Handle               │ Block 0│ Queue 0(0)         │ Queue 1(0)        │ Queue 2(0)        │ Queue 3(0)        │ Queue 4(0)         │
│                      │        │ Valid              │ Fake              │ Fake              │ Fake              │ Valid              │
│                      ├────────┼────────────────────┼───────────────────┼───────────────────┼───────────────────┼────────────────────┤
│                      │        │ Queue 0(1)         │ Queue 1(1)        │ Queue 2(1)        │ Queue 3(1)        │ Queue 4(1)         │
│                      ├────────┼────────────────────┼───────────────────┼───────────────────┼───────────────────┼────────────────────┤
│                      │        │ Queue 0(2)         │ Queue 1(2)        │ Queue 2(2)        │ Queue 3(2)        │ Queue 4(2)         │
│                      ├────────┼────────────────────┼───────────────────┼───────────────────┼───────────────────┼────────────────────┤
│                      │        │ Queue 0(3)         │ Queue 1(3)        │ Queue 2(3)        │ Queue 3(3)        │ Queue 4(3)         │
├──────────────────────┼────────┼────────────────────┼───────────────────┼───────────────────┼───────────────────┼────────────────────┤
│                      │ Block 1│ Queue 5            │ Queue 6           │ Queue 7           │ Queue 8           │ Queue 9            │
│                      ├────────┼────────────────────┼───────────────────┼───────────────────┼───────────────────┼────────────────────┤
│                      │        │ Queue 5            │ Queue 6           │ Queue 7           │ Queue 8           │ Queue 9            │
│                      ├────────┼────────────────────┼───────────────────┼───────────────────┼───────────────────┼────────────────────┤
│                      │        │ Queue 5            │ Queue 6           │ Queue 7           │ Queue 8           │ Queue 9            │
│                      ├────────┼────────────────────┼───────────────────┼───────────────────┼───────────────────┼────────────────────┤
│                      │        │ Queue 5            │ Queue 6           │ Queue 7           │ Queue 8           │ Queue 9            │
└──────────────────────┴────────┴────────────────────┴───────────────────┴───────────────────┴───────────────────┴────────────────────┘
```

## 函数原型<a name="section765814724715"></a>

```
__aicore__ inline uint16_t PostFakeMsg(__gm__ CubeMsgType* msg)
```

## 参数说明<a name="zh-cn_topic_0000001526206862_section129451113125413"></a>

**表 1**  接口参数说明

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
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

当前消息空间与该消息队列队首空间的地址偏移。

## 约束说明<a name="zh-cn_topic_0000001526206862_section65498832"></a>

无

## 调用示例<a name="zh-cn_topic_0000001526206862_section97001499599"></a>

```
hanndle.AssignQueue(queIdx);  
auto msgPtr = handle.AllocMessage();        // 获取消息空间指针msgPtr
auto offset = handle.PostFakeMsg(msgPtr);           // 在msgPtr指针位置，发送假消息
```

