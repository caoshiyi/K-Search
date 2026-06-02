# SetSkipMsg<a name="ZH-CN_TOPIC_0000002554344571"></a>

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

AIC跳过指定个数假消息的处理，仅在回调函数中调用。下图中Block0通过调用SetSkipMsg跳过三个假消息。

**图 1**  SetSkipMsg示意图<a name="fig1732521102711"></a>  
<!-- img2text -->
```text
                           Block 0 跳过三个假消息
                  ╭───────────────────────────────────────╮
                  ╰───────────────────────────────────────╯

┌────────────────┬──────────────┬──────────────┬──────────────┬──────────────┬──────────────┐
│                │   Queue 0(0) │   Queue 1(0) │   Queue 2(0) │   Queue 3(0) │   Queue 4(0) │
│                │    Valid     │     Fake     │     Fake     │     Fake     │    Valid     │
│                ├──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│                │   Queue 0(1) │   Queue 1(1) │   Queue 2(1) │   Queue 3(1) │   Queue 4(1) │
│                ├──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ CubeResGroup   │   Queue 0(2) │   Queue 1(2) │   Queue 2(2) │   Queue 3(2) │   Queue 4(2) │
│ Handle         ├──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│                │   Queue 0(3) │   Queue 1(3) │   Queue 2(3) │   Queue 3(3) │   Queue 4(3) │
│                ├──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│                │              │   Queue 5    │   Queue 6    │   Queue 7    │   Queue 8    │
│                │   Block 0    │              │              │              │              │
│                ├──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│                │              │   Queue 5    │   Queue 6    │   Queue 7    │   Queue 8    │
│                │   Block 1    │              │              │              │              │
│                ├──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│                │              │   Queue 5    │   Queue 6    │   Queue 7    │   Queue 8    │
│                │              │              │              │              │              │
│                ├──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│                │              │   Queue 5    │   Queue 6    │   Queue 7    │   Queue 8    │
│                │              │              │              │              │              │
└────────────────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────────┘
```

说明:
- 图中左侧为 `CubeResGroup Handle`。
- 上半部分是 `Block 0`，包含：
  - `Queue 0(0) Valid`
  - `Queue 1(0) Fake`
  - `Queue 2(0) Fake`
  - `Queue 3(0) Fake`
  - `Queue 4(0) Valid`
  - 以及后续三行：
    - `Queue 0(1)` ~ `Queue 4(1)`
    - `Queue 0(2)` ~ `Queue 4(2)`
    - `Queue 0(3)` ~ `Queue 4(3)`
- 上方括注 `Block 0 跳过三个假消息` 对应 `Queue 1(0)`、`Queue 2(0)`、`Queue 3(0)` 这三个 `Fake` 消息。
- 下半部分是 `Block 1`，4 行内容均为：
  - `Queue 5`、`Queue 6`、`Queue 7`、`Queue 8`、`Queue 9`。

## 函数原型<a name="section765814724715"></a>

```
 __aicore__ inline void SetSkipMsg(uint8_t skipCnt)
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
<tbody><tr id="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_row152234713443"><td class="cellrowborder" valign="top" width="9.09%" headers="mcps1.2.4.1.1 "><p id="p0731542025"><a name="p0731542025"></a><a name="p0731542025"></a>skipCnt</p>
</td>
<td class="cellrowborder" valign="top" width="7.93%" headers="mcps1.2.4.1.2 "><p id="p10731044217"><a name="p10731044217"></a><a name="p10731044217"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="82.98%" headers="mcps1.2.4.1.3 "><p id="p373442029"><a name="p373442029"></a><a name="p373442029"></a>AIC需要跳过的消息数。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无。

## 约束说明<a name="zh-cn_topic_0000001526206862_section65498832"></a>

该任务的消息空间后skipCnt个消息队列需要发送FAKE消息。

## 调用示例<a name="zh-cn_topic_0000001526206862_section97001499599"></a>

```
 __aicore__ inline static void Call(
    MatmulApiCfg &mm, __gm__ CubeMsgBody *rcvMsg, CubeResGroupHandle<CubeMsgBody> &handle)
{
    //  AIC上计算逻辑，用户自行实现
    auto skipNum = 3;//(rcvMsg->head).skipCnt，假消息个数可由用户在回调计算结构体中定义，也可以通过自定义消息结构体传递。
    auto tmpId = handle.FreeMessage(rcvMsg, AscendC::CubeMsgState::VALID);    // 当前消息处理完，调用FreeMessage，代表rcvMsg已处理完
    for (int i = 1; i < skipNum + 1; i++) {  
         // 由于后续发了三个假消息，也需要调用FreeMessage，代表假消息处理完毕。                              
         auto tmpId = handle.FreeMessage(rcvMsg + i, AscendC::CubeMsgState::FAKE);
    }
    // 当假消息存在，需要调用SetSkipMsg，通知Cube核不去处理后面三个假消息。
    handle.SetSkipMsg(skipNum);
};
```

