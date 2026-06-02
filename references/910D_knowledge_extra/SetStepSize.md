# SetStepSize<a name="ZH-CN_TOPIC_0000002523344460"></a>

## 功能说明<a name="section618mcpsimp"></a>

设置[细粒度通信](Iterate-120.md)时，通信算法的步长，即设置细粒度通信时，一次子通信任务执行或准备执行的通信算法的步骤数。例如，[图1 使用pairwise算法的AlltoAllV通信步骤示意图](Iterate-120.md#fig111591359143)中，该细粒度通信场景下，AlltoAllV通信任务的通信步长为1。

## 函数原型<a name="section620mcpsimp"></a>

```
uint32_t SetStepSize(uint8_t stepSize)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  参数说明

<a name="table9646134355611"></a>
<table><thead align="left"><tr id="row964714433565"><th class="cellrowborder" valign="top" width="14.99%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="12.02%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="72.99%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row106481443135617"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p167361341213"><a name="p167361341213"></a><a name="p167361341213"></a>stepSize</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p137362417119"><a name="p137362417119"></a><a name="p137362417119"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p165151033101217"><a name="p165151033101217"></a><a name="p165151033101217"></a>设置细粒度通信时，每次通信的步长。0表示当前非细粒度通信。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

-   0表示设置成功。
-   非0表示设置失败。

## 调用示例<a name="section1665082013318"></a>

```
static ge::graphStatus AllToAllVCustomTilingFunc(gert::TilingContext *context)
{
    AllToAllVCustomV3TilingData *tiling = context->GetTilingData<AllToAllVCustomV3TilingData>();
    const std::string groupName = "testGroup";
    const std::string algConfig = "AlltoAll=level0:fullmesh;level1:pairwise";
    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(groupName, HCCL_CMD_ALLTOALLV, algConfig, 0);
    mc2CcTilingConfig.SetStepSize(1U);
    mc2CcTilingConfig.GetTiling(tiling->mc2InitTiling);
    mc2CcTilingConfig.GetTiling(tiling->mc2CcTiling);
    return ge::GRAPH_SUCCESS;
}
```

