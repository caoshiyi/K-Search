# QuietNaN<a name="ZH-CN_TOPIC_0000002554423745"></a>

## 产品支持情况<a name="section1550532418810"></a>

<a name="zh-cn_topic_0000002523304690_table38301303189"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000002523304690_row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0000002523304690_p1883113061818"><a name="zh-cn_topic_0000002523304690_p1883113061818"></a><a name="zh-cn_topic_0000002523304690_p1883113061818"></a><span id="zh-cn_topic_0000002523304690_ph20833205312295"><a name="zh-cn_topic_0000002523304690_ph20833205312295"></a><a name="zh-cn_topic_0000002523304690_ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="zh-cn_topic_0000002523304690_p783113012187"><a name="zh-cn_topic_0000002523304690_p783113012187"></a><a name="zh-cn_topic_0000002523304690_p783113012187"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000002523304690_row1272474920205"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000002523304690_p12300735171314"><a name="zh-cn_topic_0000002523304690_p12300735171314"></a><a name="zh-cn_topic_0000002523304690_p12300735171314"></a><span id="zh-cn_topic_0000002523304690_ph730011352138"><a name="zh-cn_topic_0000002523304690_ph730011352138"></a><a name="zh-cn_topic_0000002523304690_ph730011352138"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000002523304690_p37256491200"><a name="zh-cn_topic_0000002523304690_p37256491200"></a><a name="zh-cn_topic_0000002523304690_p37256491200"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section618mcpsimp"></a>

返回指定数据类型的安静NaN值（浮点尾数最高位为1）。

Ascend 950PR/Ascend 950DT，支持的数据类型为：half/bfloat16\_t/float。

## 函数原型<a name="section620mcpsimp"></a>

-   标量接口，返回值为标量

    ```
    constexpr __aicore__ static inline T QuietNaN()
    ```

-   矢量接口，为dstLocal前count个元素赋安静NaN值

    ```
    __aicore__ static inline void QuietNaN(const LocalTensor<T>& dstLocal, uint32_t count)
    ```

## 参数说明<a name="section8793153401920"></a>

**表 1**  参数说明

<a name="table1055216132132"></a>
<table><thead align="left"><tr id="row105531513121315"><th class="cellrowborder" valign="top" width="16.49%" id="mcps1.2.4.1.1"><p id="p5553171319138"><a name="p5553171319138"></a><a name="p5553171319138"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="11.93%" id="mcps1.2.4.1.2"><p id="p5553151313131"><a name="p5553151313131"></a><a name="p5553151313131"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="71.58%" id="mcps1.2.4.1.3"><p id="p655316136139"><a name="p655316136139"></a><a name="p655316136139"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row5553201314135"><td class="cellrowborder" valign="top" width="16.49%" headers="mcps1.2.4.1.1 "><p id="p8553813111314"><a name="p8553813111314"></a><a name="p8553813111314"></a>dstLocal</p>
</td>
<td class="cellrowborder" valign="top" width="11.93%" headers="mcps1.2.4.1.2 "><p id="p755318134134"><a name="p755318134134"></a><a name="p755318134134"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="71.58%" headers="mcps1.2.4.1.3 "><p id="p1515191511407"><a name="p1515191511407"></a><a name="p1515191511407"></a>目的操作数。</p>
<p id="p65530137137"><a name="p65530137137"></a><a name="p65530137137"></a><span id="ph173308471594"><a name="ph173308471594"></a><a name="ph173308471594"></a><span id="ph9902231466"><a name="ph9902231466"></a><a name="ph9902231466"></a><span id="ph1782115034816"><a name="ph1782115034816"></a><a name="ph1782115034816"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p37511234195317"><a name="p37511234195317"></a><a name="p37511234195317"></a><span id="ph19174141065411"><a name="ph19174141065411"></a><a name="ph19174141065411"></a>LocalTensor的起始地址需要32字节对齐。</span></p>
</td>
</tr>
<tr id="row103840207421"><td class="cellrowborder" valign="top" width="16.49%" headers="mcps1.2.4.1.1 "><p id="p11183182720428"><a name="p11183182720428"></a><a name="p11183182720428"></a>count</p>
</td>
<td class="cellrowborder" valign="top" width="11.93%" headers="mcps1.2.4.1.2 "><p id="p2183122716423"><a name="p2183122716423"></a><a name="p2183122716423"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.58%" headers="mcps1.2.4.1.3 "><p id="p20183172714422"><a name="p20183172714422"></a><a name="p20183172714422"></a>输入数据元素个数。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section195571628152511"></a>

标量接口返回值为对应数据类型的安静NaN值。

矢量接口无返回值。

## 约束说明<a name="section135735192519"></a>

无。

## 调用示例<a name="section821903862513"></a>

-   标量接口

    ```
    float value = AscendC::NumericLimits<float>::QuietNaN();
    ```

-   矢量接口

    ```
    AscendC::NumericLimits<float>::QuietNaN(dstLocal, 256);
    ```

