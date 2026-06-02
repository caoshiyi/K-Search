# SetSingleOutputShape<a name="ZH-CN_TOPIC_0000002523343590"></a>

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

## 功能说明<a name="section8600561865"></a>

设置单核上结果矩阵Output的形状。

Conv3D高阶API目前支持M合轴模式的输出方式。在M合轴模式下，Conv3D API内部将Wout和Hout视为同一轴处理，输出时先沿Wout方向输出，完成一整行Wout输出后，再进行下一行的Wout输出。

**图 1**  M合轴模式示意图<a name="fig151092569912"></a>  
<!-- img2text -->
```text
                             Wout
                ┌──────────────────────────────────┐
                │  ─────────────────────────────→  │
                │   ↙────────────────────────────  │   ⎫
                │  ─────────────────────────────→  │   ⎬ 第一次输出块
Hout            │   ↙────────────────────────────  │   ⎭
                │  ─────────────────────────────→  │
                │  ↙────────────────────────────   │   ⎫
                │  ─────────────────────────────→  │   ⎬ 第二次输出块
                │   ↙───────────────────────────   │   ⎭
                └──────────────────────────────────┘

                                 M合轴模式
```

## 函数原型<a name="section9696201315613"></a>

```
__aicore__ inline void SetSingleOutputShape(uint64_t singleCo, uint64_t singleDo, uint64_t singleM)
```

## 参数说明<a name="section1071317211065"></a>

<a name="table279713116434"></a>
<table><thead align="left"><tr id="row1384217111434"><th class="cellrowborder" valign="top" width="25.562556255625562%" id="mcps1.1.4.1.1"><p id="p13842101112431"><a name="p13842101112431"></a><a name="p13842101112431"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="21.992199219921993%" id="mcps1.1.4.1.2"><p id="p784281184316"><a name="p784281184316"></a><a name="p784281184316"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="52.44524452445244%" id="mcps1.1.4.1.3"><p id="p1184241144317"><a name="p1184241144317"></a><a name="p1184241144317"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row1284281118431"><td class="cellrowborder" valign="top" width="25.562556255625562%" headers="mcps1.1.4.1.1 "><p id="p1484281124317"><a name="p1484281124317"></a><a name="p1484281124317"></a>singleCo</p>
</td>
<td class="cellrowborder" valign="top" width="21.992199219921993%" headers="mcps1.1.4.1.2 "><p id="p484281174314"><a name="p484281174314"></a><a name="p484281174314"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="52.44524452445244%" headers="mcps1.1.4.1.3 "><p id="p198421011174316"><a name="p198421011174316"></a><a name="p198421011174316"></a>单核上Output的C维度大小。</p>
</td>
</tr>
<tr id="row584221118435"><td class="cellrowborder" valign="top" width="25.562556255625562%" headers="mcps1.1.4.1.1 "><p id="p484213113437"><a name="p484213113437"></a><a name="p484213113437"></a>singleDo</p>
</td>
<td class="cellrowborder" valign="top" width="21.992199219921993%" headers="mcps1.1.4.1.2 "><p id="p58424112439"><a name="p58424112439"></a><a name="p58424112439"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="52.44524452445244%" headers="mcps1.1.4.1.3 "><p id="p184271115430"><a name="p184271115430"></a><a name="p184271115430"></a>单核上Output的D维度大小。</p>
</td>
</tr>
<tr id="row17842121184319"><td class="cellrowborder" valign="top" width="25.562556255625562%" headers="mcps1.1.4.1.1 "><p id="p1384211124311"><a name="p1384211124311"></a><a name="p1384211124311"></a>singleM</p>
</td>
<td class="cellrowborder" valign="top" width="21.992199219921993%" headers="mcps1.1.4.1.2 "><p id="p584291114312"><a name="p584291114312"></a><a name="p584291114312"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="52.44524452445244%" headers="mcps1.1.4.1.3 "><p id="p984291174316"><a name="p984291174316"></a><a name="p984291174316"></a>单核上Output的M维度大小，即H维度大小与W维度大小的乘积。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section114112291265"></a>

无

## 约束说明<a name="section121058451068"></a>

本接口当前仅支持设置Output的C维度、D维度和M维度（即H轴、W轴合并后的维度），不支持设置原始Output的大小。

## 调用示例<a name="section1552310537620"></a>

```
conv3dApi.SetSingleOutputShape(singleCoreCout, singleCoreDout, singleCoreM);
```

