# SetFlag/WaitFlag\(ISASI\)<a name="ZH-CN_TOPIC_0000002554423677"></a>

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

同一核内不同流水之间的同步指令。具有数据依赖的不同流水指令之间需要插此同步。

## 函数原型<a name="section620mcpsimp"></a>

```
template <HardEvent event>
__aicore__ inline void SetFlag(int32_t eventID)
template <HardEvent event>
__aicore__ inline void WaitFlag(int32_t eventID)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  参数说明

<a name="zh-cn_topic_0235751031_table33761356"></a>
<table><thead align="left"><tr id="zh-cn_topic_0235751031_row27598891"><th class="cellrowborder" valign="top" width="18.54%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="10.05%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="71.41%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row42461942101815"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p479605232211"><a name="p479605232211"></a><a name="p479605232211"></a>event</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p1579635215228"><a name="p1579635215228"></a><a name="p1579635215228"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p718711376505"><a name="p718711376505"></a><a name="p718711376505"></a>模板参数。</p>
<p id="p1179555214221"><a name="p1179555214221"></a><a name="p1179555214221"></a>同步事件，数据类型为HardEvent。详细内容参考下文中的同步类型说明。</p>
</td>
</tr>
<tr id="row2137145181815"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p179035252218"><a name="p179035252218"></a><a name="p179035252218"></a>eventID</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p7789185214226"><a name="p7789185214226"></a><a name="p7789185214226"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p13635175184918"><a name="p13635175184918"></a><a name="p13635175184918"></a>事件ID。数据类型为int32_t类型。其定义如下：</p>
<p id="p1140180105210"><a name="p1140180105210"></a><a name="p1140180105210"></a>在基于TPipe和TQue编程场景中，eventID需要通过<a href="AllocEventID.md">AllocEventID</a>或者<a href="FetchEventID.md">FetchEventID</a>来获取。</p>
<p id="p468305719192"><a name="p468305719192"></a><a name="p468305719192"></a><span id="ph126252025205"><a name="ph126252025205"></a><a name="ph126252025205"></a>Ascend 950PR/Ascend 950DT</span>，数据范围为：0-7</p>
</td>
</tr>
</tbody>
</table>

同步类型说明如下:

```
enum class HardEvent : uint8_t {
    // 名称（源流水_目标流水），例如MTE2_V，代表PIPE_MTE2为源流水，PIPE_V为目标流水。标识从PIPE_MTE2到PIPE_V的同步，PIPE_V等待PIPE_MTE2。
    MTE2_MTE1
    MTE1_MTE2
    MTE1_M
    M_MTE1
    MTE2_V
    V_MTE2
    MTE3_V
    V_MTE3
    M_V
    V_M
    V_V
    MTE3_MTE1
    MTE1_MTE3
    MTE1_V
    MTE2_M
    M_MTE2
    V_MTE1
    M_FIX
    FIX_M
    MTE3_MTE2
    MTE2_MTE3
    S_V
    V_S
    S_MTE2
    MTE2_S
    S_MTE3
    MTE3_S
    MTE2_FIX
    FIX_MTE2
    FIX_S
    M_S
    FIX_MTE3
}
```

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

-   SetFlag/WaitFlag必须成对出现。
-   在基于TPipe和TQue编程场景中，禁止用户在使用SetFlag和WaitFlag时自行指定eventID，容易与框架同步事件冲突，导致卡死问题。
-   在静态Tensor编程场景中，事件的类型和事件ID由开发者自行管理，但需要注意事件ID不能使用6和7（可能与内部使用的事件ID出现冲突，进而出现未定义行为）。

## 调用示例<a name="section837496171220"></a>

如DataCopy需要等待SetValue执行完成后才能执行，需要插入PIPE\_S到PIPE\_MTE3的同步。

```
AscendC::GlobalTensor<half> dstGlobal;
AscendC::LocalTensor<half> dstLocal;
dstLocal.SetValue(0, 0);
uint32_t dataSize = 512;
// 基于TPipe和TQue编程场景中，eventID需要通过AllocEventID或FetchEventID获取
int32_t eventIDSToMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_MTE3));
AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(eventIDSToMTE3);
AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(eventIDSToMTE3);
// 静态Tensor编程场景中，eventID由开发者自行管理
// AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
// AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
AscendC::DataCopy(dstGlobal, dstLocal, dataSize);
```

