# GetTensorC<a name="ZH-CN_TOPIC_0000002554424119"></a>

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

## 功能说明<a name="section618mcpsimp"></a>

在完成Iterate操作后调用本接口，获取结果矩阵块，完成数据从L0C到GM的搬运。此接口与[Iterate](Iterate-132.md)接口配合使用，用于在Iterate执行迭代计算后，获取结果矩阵。

## 函数原型<a name="section620mcpsimp"></a>

```
template <bool sync = true>
__aicore__ inline void GetTensorC(const AscendC::GlobalTensor<DstT> &output, uint8_t enAtomic = 0, bool enSequentialWrite = false)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table8746171282418"></a>
<table><thead align="left"><tr id="row8746191212419"><th class="cellrowborder" valign="top" width="17.04%" id="mcps1.2.3.1.1"><p id="p474618126245"><a name="p474618126245"></a><a name="p474618126245"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="82.96%" id="mcps1.2.3.1.2"><p id="p1574681216240"><a name="p1574681216240"></a><a name="p1574681216240"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row1574617127244"><td class="cellrowborder" valign="top" width="17.04%" headers="mcps1.2.3.1.1 "><p id="p2746171214244"><a name="p2746171214244"></a><a name="p2746171214244"></a>sync</p>
</td>
<td class="cellrowborder" valign="top" width="82.96%" headers="mcps1.2.3.1.2 "><p id="p12551641162414"><a name="p12551641162414"></a><a name="p12551641162414"></a>预留参数，用户无需感知。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  接口参数说明

<a name="zh-cn_topic_0235751031_table33761356"></a>
<table><thead align="left"><tr id="zh-cn_topic_0235751031_row27598891"><th class="cellrowborder" valign="top" width="19.220000000000002%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="11.690000000000001%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="69.08999999999999%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row42461942101815"><td class="cellrowborder" valign="top" width="19.220000000000002%" headers="mcps1.2.4.1.1 "><p id="p19866193819473"><a name="p19866193819473"></a><a name="p19866193819473"></a>output</p>
</td>
<td class="cellrowborder" valign="top" width="11.690000000000001%" headers="mcps1.2.4.1.2 "><p id="p33487148556"><a name="p33487148556"></a><a name="p33487148556"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="69.08999999999999%" headers="mcps1.2.4.1.3 "><p id="p246271553317"><a name="p246271553317"></a><a name="p246271553317"></a>将计算结果搬至Global Memory的GM地址。</p>
</td>
</tr>
<tr id="row165215337110"><td class="cellrowborder" valign="top" width="19.220000000000002%" headers="mcps1.2.4.1.1 "><p id="p75312334116"><a name="p75312334116"></a><a name="p75312334116"></a>enAtomic</p>
</td>
<td class="cellrowborder" valign="top" width="11.690000000000001%" headers="mcps1.2.4.1.2 "><p id="p9531133191113"><a name="p9531133191113"></a><a name="p9531133191113"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="69.08999999999999%" headers="mcps1.2.4.1.3 "><p id="p16531933131118"><a name="p16531933131118"></a><a name="p16531933131118"></a>预留参数，用户无需感知。</p>
</td>
</tr>
<tr id="row424218353119"><td class="cellrowborder" valign="top" width="19.220000000000002%" headers="mcps1.2.4.1.1 "><p id="p152421635131111"><a name="p152421635131111"></a><a name="p152421635131111"></a>enSequentialWrite</p>
</td>
<td class="cellrowborder" valign="top" width="11.690000000000001%" headers="mcps1.2.4.1.2 "><p id="p53958383125"><a name="p53958383125"></a><a name="p53958383125"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="69.08999999999999%" headers="mcps1.2.4.1.3 "><p id="p524263510117"><a name="p524263510117"></a><a name="p524263510117"></a>预留参数，用户无需感知。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

GetTensorC接口必须在Iterate后进行调用，完成卷积反向实现，调用顺序如下。

```
while (Iterate()) {   
    GetTensorC(); 
}
```

## 调用示例<a name="section1665082013318"></a>

```
while (gradInput_.Iterate()) {   
    gradInput_.GetTensorC(gradInputGm_[offsetC_]); 
}
```

