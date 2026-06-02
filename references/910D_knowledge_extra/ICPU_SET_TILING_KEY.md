# ICPU\_SET\_TILING\_KEY<a name="ZH-CN_TOPIC_0000002554423707"></a>

## 功能说明<a name="zh-cn_topic_0000002000279997_zh-cn_topic_0000001610028277_section259105813316"></a>

用于指定本次CPU调测使用的tilingKey。调测执行时，将只执行算子核函数中该tilingKey对应的分支。

## 函数原型<a name="zh-cn_topic_0000002000279997_zh-cn_topic_0000001610028277_section2067518173415"></a>

```
ICPU_SET_TILING_KEY(tilingKey)
```

## 参数说明<a name="zh-cn_topic_0000002000279997_zh-cn_topic_0000001610028277_section158061867342"></a>

<a name="zh-cn_topic_0000002000279997_zh-cn_topic_0000001610028277_zh-cn_topic_0235751031_table33761356"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000002000279997_zh-cn_topic_0000001610028277_zh-cn_topic_0235751031_row27598891"><th class="cellrowborder" valign="top" width="16.49%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000002000279997_zh-cn_topic_0000001610028277_zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0000002000279997_zh-cn_topic_0000001610028277_zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0000002000279997_zh-cn_topic_0000001610028277_zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="11.93%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000002000279997_zh-cn_topic_0000001610028277_zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0000002000279997_zh-cn_topic_0000001610028277_zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0000002000279997_zh-cn_topic_0000001610028277_zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="71.58%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000002000279997_zh-cn_topic_0000001610028277_zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0000002000279997_zh-cn_topic_0000001610028277_zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0000002000279997_zh-cn_topic_0000001610028277_zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000002000279997_zh-cn_topic_0000001610028277_row42461942101815"><td class="cellrowborder" valign="top" width="16.49%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002000279997_zh-cn_topic_0000001610028277_p811762918540"><a name="zh-cn_topic_0000002000279997_zh-cn_topic_0000001610028277_p811762918540"></a><a name="zh-cn_topic_0000002000279997_zh-cn_topic_0000001610028277_p811762918540"></a>tilingKey</p>
</td>
<td class="cellrowborder" valign="top" width="11.93%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002000279997_zh-cn_topic_0000001610028277_p211672965416"><a name="zh-cn_topic_0000002000279997_zh-cn_topic_0000001610028277_p211672965416"></a><a name="zh-cn_topic_0000002000279997_zh-cn_topic_0000001610028277_p211672965416"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.58%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002000279997_zh-cn_topic_0000001610028277_p131151729165417"><a name="zh-cn_topic_0000002000279997_zh-cn_topic_0000001610028277_p131151729165417"></a><a name="zh-cn_topic_0000002000279997_zh-cn_topic_0000001610028277_p131151729165417"></a>指定本次CPU调测使用的tilingKey，参数类型为int32_t。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0000002000279997_zh-cn_topic_0000001610028277_section640mcpsimp"></a>

无

## 约束说明<a name="zh-cn_topic_0000002000279997_zh-cn_topic_0000001610028277_section794123819592"></a>

-   未使用该接口设置tilingKey的情况下，tilingKey将会为默认值0，在调测执行时，会有告警提示Tiling Key是0，并继续进行调测。如果核函数中有tilingKey分支，将会执行tilingKey为0的分支，其他tilingKey对应的分支不会执行。
-   tilingKey建议传入正整数，如果设置为负数或者0，将会告警并继续调测。如果传入0，将会执行tilingKey为0的分支；tilingKey传入负数，将导致未定义的行为。
-   该接口需要在[ICPU\_RUN\_KF](ICPU_RUN_KF.md)前调用。

## 调用示例<a name="zh-cn_topic_0000002000279997_zh-cn_topic_0000001610028277_section82241477610"></a>

```
ICPU_SET_TILING_KEY(10086);
ICPU_RUN_KF(sort_kernel0, coreNum, (uint8_t*)x, (uint8_t*)y);
```

