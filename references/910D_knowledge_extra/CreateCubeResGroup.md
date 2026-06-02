# CreateCubeResGroup<a name="ZH-CN_TOPIC_0000002554423511"></a>

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

快速创建CubeResGroupHandle对象，内部完成消息队列空间和同步事件分配。推荐使用该接口，避免使用CubeResGroupHandle的构造函数创建对象，出现不同对象的消息队列空间冲突、同步事件错误等情况。

## 函数原型<a name="section765814724715"></a>

```
template <int groupID, class MatmulApiType, template <class, class> class CallBack, typename CubeMsgType>
__aicore__ inline CubeResGroupHandle<CubeMsgType> CreateCubeResGroup(KfcWorkspace& desc, uint8_t blockStart, uint8_t blockSize, uint8_t msgQueueSize, GM_ADDR tiling)
```

## 参数说明<a name="zh-cn_topic_0000001526206862_section129451113125413"></a>

**表 1**  模板参数说明

<a name="table775915013250"></a>
<table><thead align="left"><tr id="row18759203252"><th class="cellrowborder" valign="top" width="13.270000000000001%" id="mcps1.2.3.1.1"><p id="p175919042514"><a name="p175919042514"></a><a name="p175919042514"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="86.72999999999999%" id="mcps1.2.3.1.2"><p id="p1759150132516"><a name="p1759150132516"></a><a name="p1759150132516"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row187601006257"><td class="cellrowborder" valign="top" width="13.270000000000001%" headers="mcps1.2.3.1.1 "><p id="p147601503258"><a name="p147601503258"></a><a name="p147601503258"></a>groupID</p>
</td>
<td class="cellrowborder" valign="top" width="86.72999999999999%" headers="mcps1.2.3.1.2 "><p id="p157601405251"><a name="p157601405251"></a><a name="p157601405251"></a>用于表示Group的编号，int32取值范围。</p>
</td>
</tr>
<tr id="row67608072518"><td class="cellrowborder" valign="top" width="13.270000000000001%" headers="mcps1.2.3.1.1 "><p id="p177607013255"><a name="p177607013255"></a><a name="p177607013255"></a>MatmulApiType</p>
</td>
<td class="cellrowborder" valign="top" width="86.72999999999999%" headers="mcps1.2.3.1.2 "><p id="p3760604250"><a name="p3760604250"></a><a name="p3760604250"></a>定义的AIC计算对象类型。</p>
</td>
</tr>
<tr id="row11760306250"><td class="cellrowborder" valign="top" width="13.270000000000001%" headers="mcps1.2.3.1.1 "><p id="p167601302256"><a name="p167601302256"></a><a name="p167601302256"></a>CallBack</p>
</td>
<td class="cellrowborder" valign="top" width="86.72999999999999%" headers="mcps1.2.3.1.2 "><p id="p167608022510"><a name="p167608022510"></a><a name="p167608022510"></a>回调函数类，需要实现Init和Call两个接口。</p>
</td>
</tr>
<tr id="row187601209251"><td class="cellrowborder" valign="top" width="13.270000000000001%" headers="mcps1.2.3.1.1 "><p id="p1776020012258"><a name="p1776020012258"></a><a name="p1776020012258"></a>CubeMsgType</p>
</td>
<td class="cellrowborder" valign="top" width="86.72999999999999%" headers="mcps1.2.3.1.2 "><p id="p18760160162512"><a name="p18760160162512"></a><a name="p18760160162512"></a>用户自定义的消息结构体。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  接口参数说明

<a name="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_table111938719446"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_row6223476444"><th class="cellrowborder" valign="top" width="12.97%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p10223674448"><a name="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p10223674448"></a><a name="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p10223674448"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="8.110000000000001%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p645511218169"><a name="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p645511218169"></a><a name="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p645511218169"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="78.92%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p1922337124411"><a name="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p1922337124411"></a><a name="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p1922337124411"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_row152234713443"><td class="cellrowborder" valign="top" width="12.97%" headers="mcps1.2.4.1.1 "><p id="p1060615396919"><a name="p1060615396919"></a><a name="p1060615396919"></a>desc</p>
</td>
<td class="cellrowborder" valign="top" width="8.110000000000001%" headers="mcps1.2.4.1.2 "><p id="p13606163918914"><a name="p13606163918914"></a><a name="p13606163918914"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="78.92%" headers="mcps1.2.4.1.3 "><p id="p1560613391196"><a name="p1560613391196"></a><a name="p1560613391196"></a><a href="KfcWorkspace.md">KfcWorkspace</a>，用于维护消息队列空间。</p>
</td>
</tr>
<tr id="row519512376588"><td class="cellrowborder" valign="top" width="12.97%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001526206862_p223953193015"><a name="zh-cn_topic_0000001526206862_p223953193015"></a><a name="zh-cn_topic_0000001526206862_p223953193015"></a>blockStart</p>
</td>
<td class="cellrowborder" valign="top" width="8.110000000000001%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001526206862_p7239938308"><a name="zh-cn_topic_0000001526206862_p7239938308"></a><a name="zh-cn_topic_0000001526206862_p7239938308"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="78.92%" headers="mcps1.2.4.1.3 "><p id="p1117515154210"><a name="p1117515154210"></a><a name="p1117515154210"></a>该CubeResGroupHandle在AIV视角下的起始AIC对应的序号，即AIC起始序号 * 2。例如，如果AIC起始序号为0，则填入0 * 2；如果为1，则填入1 * 2。</p>
</td>
</tr>
<tr id="row11725173915582"><td class="cellrowborder" valign="top" width="12.97%" headers="mcps1.2.4.1.1 "><p id="p11374343181311"><a name="p11374343181311"></a><a name="p11374343181311"></a>blockSize</p>
</td>
<td class="cellrowborder" valign="top" width="8.110000000000001%" headers="mcps1.2.4.1.2 "><p id="p146153901420"><a name="p146153901420"></a><a name="p146153901420"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="78.92%" headers="mcps1.2.4.1.3 "><p id="p517518151222"><a name="p517518151222"></a><a name="p517518151222"></a>该CubeResGroupHandle在AIV视角下分配的Block个数，即实际的AIC个数*2。</p>
</td>
</tr>
<tr id="row41174317586"><td class="cellrowborder" valign="top" width="12.97%" headers="mcps1.2.4.1.1 "><p id="p15285194613135"><a name="p15285194613135"></a><a name="p15285194613135"></a>msgQueueSize</p>
</td>
<td class="cellrowborder" valign="top" width="8.110000000000001%" headers="mcps1.2.4.1.2 "><p id="p9285104615133"><a name="p9285104615133"></a><a name="p9285104615133"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="78.92%" headers="mcps1.2.4.1.3 "><p id="p101756155210"><a name="p101756155210"></a><a name="p101756155210"></a>该CubeResGroupHandle分配的消息队列总数。</p>
</td>
</tr>
<tr id="row20135231122916"><td class="cellrowborder" valign="top" width="12.97%" headers="mcps1.2.4.1.1 "><p id="p176065399919"><a name="p176065399919"></a><a name="p176065399919"></a>tiling</p>
</td>
<td class="cellrowborder" valign="top" width="8.110000000000001%" headers="mcps1.2.4.1.2 "><p id="p96061039494"><a name="p96061039494"></a><a name="p96061039494"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="78.92%" headers="mcps1.2.4.1.3 "><p id="p5606039191"><a name="p5606039191"></a><a name="p5606039191"></a>AIC核计算所需tiling信息的地址。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

CubeResGroupHandle对象实例。

## 约束说明<a name="section84021317123613"></a>

-   假设芯片的AIV核数为x，那么blockStart + blockSize <= x - 1, msgQueueSize <= x。
-   每个AIC至少被分配1个msgQueue。
-   blockStart和blockSize必须为偶数。
-   使用该接口，UB空间末尾的1600B + sizeof\(CubeMsgType\)将被占用。
-   1个AIC只能属于1个CubeGroupHandle，即多个CubeGroupHandle的\[blockStart / 2, blockStart / 2 + blockSize / 2\]区间不能重叠。
-   不能和[REGIST\_MATMUL\_OBJ](REGIST_MATMUL_OBJ.md)接口同时使用。使用资源管理API时，用户自主管理AIC和AIV的核间通信，REGIST\_MATMUL\_OBJ内部是由框架管理AIC和AIV的核间通信，同时使用可能会导致通信消息错误等异常。

## 调用示例<a name="zh-cn_topic_0000001526206862_section97001499599"></a>

```
auto handle = AscendC::CreateCubeResGroup<GROUPID, MatmulApiType, MyCallbackFunc, CubeMsgBody> (desc, BLOCKSTART, BLOCKSIZE, MSGQUEUESIZE, tilingGM);
```

