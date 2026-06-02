# SetCcTiling（废弃）<a name="ZH-CN_TOPIC_0000002554423555"></a>

> **说明：** 
>该接口废弃，并将在后续版本移除，请不要使用该接口。请使用[SetCcTilingV2](SetCcTilingV2.md)接口设置通信算法的Tiling地址。

## 产品支持情况<a name="section1586581915393"></a>

<a name="table169596713360"></a>
<table><thead align="left"><tr id="row129590715369"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p17959971362"><a name="p17959971362"></a><a name="p17959971362"></a><span id="ph895914718367"><a name="ph895914718367"></a><a name="ph895914718367"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p89594763612"><a name="p89594763612"></a><a name="p89594763612"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row18959673369"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p1595910763613"><a name="p1595910763613"></a><a name="p1595910763613"></a><span id="ph1595918753613"><a name="ph1595918753613"></a><a name="ph1595918753613"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p1695957133611"><a name="p1695957133611"></a><a name="p1695957133611"></a>x</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section1769414212182"></a>

用于设置HCCL客户端通信算法的Tiling地址。

## 函数原型<a name="section14969112112188"></a>

```
__aicore__ inline int32_t SetCcTiling(__gm__ void *ccOpTilingData)
```

## 参数说明<a name="section12546122891815"></a>

**表 1**  接口参数说明

<a name="table11541249132419"></a>
<table><thead align="left"><tr id="row81541849152411"><th class="cellrowborder" valign="top" width="15.981598159815983%" id="mcps1.2.4.1.1"><p id="p715444932416"><a name="p715444932416"></a><a name="p715444932416"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="19.801980198019802%" id="mcps1.2.4.1.2"><p id="p1115410497248"><a name="p1115410497248"></a><a name="p1115410497248"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="64.2164216421642%" id="mcps1.2.4.1.3"><p id="p41549495249"><a name="p41549495249"></a><a name="p41549495249"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row11541349102415"><td class="cellrowborder" valign="top" width="15.981598159815983%" headers="mcps1.2.4.1.1 "><p id="p1615414972415"><a name="p1615414972415"></a><a name="p1615414972415"></a>ccOpTilingData</p>
</td>
<td class="cellrowborder" valign="top" width="19.801980198019802%" headers="mcps1.2.4.1.2 "><p id="p1615434911249"><a name="p1615434911249"></a><a name="p1615434911249"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="64.2164216421642%" headers="mcps1.2.4.1.3 "><p id="p11154194915243"><a name="p11154194915243"></a><a name="p11154194915243"></a>通信算法的<a href="TilingData结构体.md#table678914014562">Mc2CcTiling</a>参数的地址。Mc2CcTiling在Host侧计算得出，具体请参考<a href="TilingData结构体.md#table678914014562">表2 Mc2CcTiling参数说明</a>，由框架传递到Kernel函数中使用，完整示例请参考<a href="HCCL模板参数.md#section11493459173619">8.13.1.2-调用示例</a>。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section641993511814"></a>

-   HCCL\_SUCCESS，表示成功。
-   HCCL\_FAILED，表示失败。

## 约束说明<a name="section15931595196"></a>

-   参数相同的同一种通信算法在调用Prepare接口前只需要调用一次本接口，否则需要多次调用本接口。
-   同一种通信算法只支持设置一个ccOpTilingData地址；对于同一种通信算法，重复调用本接口会覆盖该通信算法的ccOpTilingData地址。
-   若调用本接口，必须与传initTiling地址的[Init](zh-cn_topic_0000002523303562.md)接口配合使用，且Init接口在本接口前被调用。
-   若调用本接口，必须使用标准C++语法定义TilingData结构体的开发方式，具体请参考[使用标准C++语法定义Tiling结构体](使用标准C++语法定义Tiling结构体.md)。

