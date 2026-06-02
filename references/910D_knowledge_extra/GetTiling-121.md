# GetTiling<a name="ZH-CN_TOPIC_0000002523343720"></a>

## 功能说明<a name="section618mcpsimp"></a>

获取[Mc2InitTiling](TilingData结构体.md#table4835205712588)参数和[Mc2CcTiling](TilingData结构体.md#table678914014562)参数。

## 函数原型<a name="section620mcpsimp"></a>

```
uint32_t GetTiling(::Mc2InitTiling &tiling)
```

```
uint32_t GetTiling(::Mc2CcTiling &tiling)
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
<tbody><tr id="row106481443135617"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p201012231161"><a name="p201012231161"></a><a name="p201012231161"></a>tiling</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p0108231864"><a name="p0108231864"></a><a name="p0108231864"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p1973172619349"><a name="p1973172619349"></a><a name="p1973172619349"></a>Tiling结构体存储的Tiling信息。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

-   返回值为0，则tiling计算成功，该Tiling结构体的值可以用于后续计算。
-   返回值非0，则tiling计算失败，该Tiling结果无法使用。

## 约束说明<a name="section633mcpsimp"></a>

无

## 调用示例<a name="section1665082013318"></a>

```
const char *groupName = "testGroup";
uint32_t opType = HCCL_CMD_REDUCE_SCATTER;
std::string algConfig = "ReduceScatter=level0:fullmesh";
uint32_t reduceType = HCCL_REDUCE_SUM;
AscendC::Mc2CcTilingConfig mc2CcTilingConfig(groupName, opType, algConfig, reduceType);
mc2CcTilingConfig.GetTiling(tiling->mc2InitTiling); // tiling为算子组装的TilingData结构体，获取Mc2InitTiling
mc2CcTilingConfig.GetTiling(tiling->reduceScatterTiling); // tiling为算子组装的TilingData结构体，获取Mc2CcTiling
```

