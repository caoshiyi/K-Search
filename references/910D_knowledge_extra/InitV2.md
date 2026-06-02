# InitV2<a name="ZH-CN_TOPIC_0000002554423991"></a>

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
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p1695957133611"><a name="p1695957133611"></a><a name="p1695957133611"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section618mcpsimp"></a>

HCCL客户端初始化接口。该接口默认在所有核上工作，用户也可以在调用前通过[GetBlockIdx](GetBlockIdx.md)指定其在某一个核上运行。

## 函数原型<a name="section620mcpsimp"></a>

```
__aicore__ inline void InitV2(GM_ADDR context, const void *initTiling)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  接口参数说明

<a name="zh-cn_topic_0235751031_table33761356"></a>
<table><thead align="left"><tr id="zh-cn_topic_0235751031_row27598891"><th class="cellrowborder" valign="top" width="17.77%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="11.799999999999999%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="70.43%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row42461942101815"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p163481714145518"><a name="p163481714145518"></a><a name="p163481714145518"></a>context</p>
</td>
<td class="cellrowborder" valign="top" width="11.799999999999999%" headers="mcps1.2.4.1.2 "><p id="p33487148556"><a name="p33487148556"></a><a name="p33487148556"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="70.43%" headers="mcps1.2.4.1.3 "><p id="p13481914175514"><a name="p13481914175514"></a><a name="p13481914175514"></a>通信上下文，包含rankDim，rankID等相关信息。通过框架提供的获取通信上下文的接口<a href="GetHcclContext.md">GetHcclContext</a>获取context。</p>
</td>
</tr>
<tr id="row71011416203015"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p51015161301"><a name="p51015161301"></a><a name="p51015161301"></a>initTiling</p>
</td>
<td class="cellrowborder" valign="top" width="11.799999999999999%" headers="mcps1.2.4.1.2 "><p id="p121011916143010"><a name="p121011916143010"></a><a name="p121011916143010"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="70.43%" headers="mcps1.2.4.1.3 "><p id="p7101111612301"><a name="p7101111612301"></a><a name="p7101111612301"></a>通信域初始化<a href="TilingData结构体.md#table4835205712588">Mc2InitTiling</a>的地址。<a href="TilingData结构体.md#table4835205712588">Mc2InitTiling</a>在Host侧计算得出，具体请参考<a href="TilingData结构体.md#table4835205712588">表1 Mc2InitTiling参数说明</a>，由框架传递到Kernel函数中使用。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

-   本接口必须与[SetCcTilingV2](SetCcTilingV2.md)接口配合使用。
-   调用本接口时，必须使用标准C++语法定义TilingData结构体的开发方式，具体请参考[使用标准C++语法定义Tiling结构体](使用标准C++语法定义Tiling结构体.md)。
-   调用本接口传入的initTiling参数，不能使用Global Memory地址，建议通过[GET\_TILING\_DATA\_WITH\_STRUCT](GET_TILING_DATA_WITH_STRUCT.md)接口获取TilingData的栈地址。
-   本接口不支持使用相同的context初始化多个HCCL对象。

## 调用示例<a name="section1665082013318"></a>

用户自定义TilingData结构体：

```
class UserCustomTilingData {
    AscendC::tiling::Mc2InitTiling initTiling;
    AscendC::tiling::Mc2CcTiling tiling;
    CustomTiling param;
};
```

在所有核上创建HCCL对象，并调用InitV2接口初始化：

```
extern "C" __global__ __aicore__ void userKernel(GM_ADDR aGM, GM_ADDR workspaceGM, GM_ADDR tilingGM) {
    REGISTER_TILING_DEFAULT(UserCustomTilingData);
    GET_TILING_DATA_WITH_STRUCT(UserCustomTilingData,tilingData,tilingGM);

    GM_ADDR contextGM = AscendC::GetHcclContext<0>(); 
    Hccl hccl;
    hccl.InitV2(contextGM, &tilingData);
    hccl.SetCcTilingV2(offsetof(UserCustomTilingData, tiling));

    // 调用HCCL的Prepare、Commit、Wait、Finalize接口
}
```

