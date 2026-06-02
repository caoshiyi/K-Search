# DeNormMin<a name="ZH-CN_TOPIC_0000002523303682"></a>

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

返回指定数据类型的最小正的非正规值。

Ascend 950PR/Ascend 950DT，支持的数据类型为：half/bfloat16\_t/float。

## 函数原型<a name="section620mcpsimp"></a>

-   标量接口，返回值为标量

    ```
    constexpr __aicore__ static inline T DeNormMin()
    ```

-   矢量接口，为dstLocal前count个元素赋最小正的非正规值

    ```
    __aicore__ static inline void DeNormMin(const LocalTensor<T>& dstLocal, uint32_t count)
    ```

## 参数说明<a name="section17119181914182"></a>

**表 1**  参数说明

<a name="table1112011192180"></a>
<table><thead align="left"><tr id="row212031921811"><th class="cellrowborder" valign="top" width="16.49%" id="mcps1.2.4.1.1"><p id="p1012051971817"><a name="p1012051971817"></a><a name="p1012051971817"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="11.93%" id="mcps1.2.4.1.2"><p id="p712013190183"><a name="p712013190183"></a><a name="p712013190183"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="71.58%" id="mcps1.2.4.1.3"><p id="p4120141913183"><a name="p4120141913183"></a><a name="p4120141913183"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row1712020193181"><td class="cellrowborder" valign="top" width="16.49%" headers="mcps1.2.4.1.1 "><p id="p14120719161820"><a name="p14120719161820"></a><a name="p14120719161820"></a>dstLocal</p>
</td>
<td class="cellrowborder" valign="top" width="11.93%" headers="mcps1.2.4.1.2 "><p id="p1612015191181"><a name="p1612015191181"></a><a name="p1612015191181"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="71.58%" headers="mcps1.2.4.1.3 "><p id="p712041910183"><a name="p712041910183"></a><a name="p712041910183"></a>目的操作数。</p>
<p id="p5120319161813"><a name="p5120319161813"></a><a name="p5120319161813"></a><span id="ph612019197187"><a name="ph612019197187"></a><a name="ph612019197187"></a><span id="ph161201319191819"><a name="ph161201319191819"></a><a name="ph161201319191819"></a><span id="ph6120141911189"><a name="ph6120141911189"></a><a name="ph6120141911189"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p1412011917189"><a name="p1412011917189"></a><a name="p1412011917189"></a><span id="ph71204195183"><a name="ph71204195183"></a><a name="ph71204195183"></a>LocalTensor的起始地址需要32字节对齐。</span></p>
</td>
</tr>
<tr id="row11120131919188"><td class="cellrowborder" valign="top" width="16.49%" headers="mcps1.2.4.1.1 "><p id="p11183182720428"><a name="p11183182720428"></a><a name="p11183182720428"></a>count</p>
</td>
<td class="cellrowborder" valign="top" width="11.93%" headers="mcps1.2.4.1.2 "><p id="p1912051981811"><a name="p1912051981811"></a><a name="p1912051981811"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.58%" headers="mcps1.2.4.1.3 "><p id="p512051916181"><a name="p512051916181"></a><a name="p512051916181"></a>输入数据元素个数。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section1212181911812"></a>

标量接口返回值为对应数据类型的最小正的非正规值。

矢量接口无返回值。

## 约束说明<a name="section0121131919189"></a>

无。

## 调用示例<a name="section121218192180"></a>

-   标量接口

    ```
    float value = AscendC::NumericLimits<float>::DeNormMin();
    ```

-   矢量接口

    ```
    AscendC::NumericLimits<float>::DeNormMin(dstLocal, 256);
    ```

