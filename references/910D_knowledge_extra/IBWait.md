# IBWait<a name="ZH-CN_TOPIC_0000002554424095"></a>

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

当不同核之间操作同一块全局内存且可能存在读后写、写后读以及写后写等数据依赖问题时，通过调用该函数来插入同步语句来避免上述数据依赖时可能出现的数据读写错误问题。IBWait与IBSet成对出现配合使用，表示核之间的同步等待指令，等待某一个核操作完成。

## 函数原型<a name="section620mcpsimp"></a>

```
template <bool isAIVOnly = true>
__aicore__ inline void IBWait(const GlobalTensor<int32_t>& gmWorkspace, const LocalTensor<int32_t>& ubWorkspace, int32_t blockIdx, int32_t eventID)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table1596943217411"></a>
<table><thead align="left"><tr id="row1596916326417"><th class="cellrowborder" valign="top" width="18.62%" id="mcps1.2.3.1.1"><p id="p17970143211417"><a name="p17970143211417"></a><a name="p17970143211417"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="81.38%" id="mcps1.2.3.1.2"><p id="p8970163210416"><a name="p8970163210416"></a><a name="p8970163210416"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row12970113214410"><td class="cellrowborder" valign="top" width="18.62%" headers="mcps1.2.3.1.1 "><p id="p9970193217417"><a name="p9970193217417"></a><a name="p9970193217417"></a>isAIVOnly</p>
</td>
<td class="cellrowborder" valign="top" width="81.38%" headers="mcps1.2.3.1.2 "><p id="p29702032748"><a name="p29702032748"></a><a name="p29702032748"></a>控制是否为AIVOnly模式，默认为true。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  接口参数说明

<a name="zh-cn_topic_0235751031_table33761356"></a>
<table><thead align="left"><tr id="zh-cn_topic_0235751031_row27598891"><th class="cellrowborder" valign="top" width="18.54%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="10.05%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="71.41%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row42461942101815"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p284425844311"><a name="p284425844311"></a><a name="p284425844311"></a>gmWorkspace</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p158449584436"><a name="p158449584436"></a><a name="p158449584436"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p3844958114318"><a name="p3844958114318"></a><a name="p3844958114318"></a>外部存储核状态的公共缓存，类型为GlobalTensor。GlobalTensor数据结构的定义请参考<a href="GlobalTensor.md">GlobalTensor</a>。</p>
</td>
</tr>
<tr id="row2137145181815"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p6844125874315"><a name="p6844125874315"></a><a name="p6844125874315"></a>ubWorkspace</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p128442058144312"><a name="p128442058144312"></a><a name="p128442058144312"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p14867058115319"><a name="p14867058115319"></a><a name="p14867058115319"></a>当前核的公共缓存。</p>
<p id="p16703131355116"><a name="p16703131355116"></a><a name="p16703131355116"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</p>
</td>
</tr>
<tr id="row19615183817191"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p1484519586432"><a name="p1484519586432"></a><a name="p1484519586432"></a>blockIdx</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p484514581433"><a name="p484514581433"></a><a name="p484514581433"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p514534616180"><a name="p514534616180"></a><a name="p514534616180"></a>表示等待核的idx号，取值范围：[0, 核数-1]，不包含自身blockIdx。</p>
</td>
</tr>
<tr id="row250163914199"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p98451586430"><a name="p98451586430"></a><a name="p98451586430"></a>eventID</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p20845205894317"><a name="p20845205894317"></a><a name="p20845205894317"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p17845145813432"><a name="p17845145813432"></a><a name="p17845145813432"></a>用来控制当前核的set、wait事件。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

-   gmWorkspace申请的空间最少要求为：核数 \* 32Bytes \* eventID\_max + blockIdx\_max \* 32Bytes + 32Bytes。（eventID\_max和blockIdx\_max分别指eventID、blockIdx的最大值 ）
-   ubWorkspace申请的空间最少要求为：32Bytes。
-   使用该接口进行多核控制时，算子调用时指定的逻辑numBlocks必须保证不大于实际运行该算子的AI处理器核数，否则框架进行多轮调度时会插入异常同步，导致Kernel“卡死”现象。

## 调用示例<a name="section177231425115410"></a>

调用样例请参考[调用示例](IBSet.md#section177231425115410)。

