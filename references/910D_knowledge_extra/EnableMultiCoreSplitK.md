# EnableMultiCoreSplitK<a name="ZH-CN_TOPIC_0000002523343698"></a>

## 功能说明<a name="section618mcpsimp"></a>

多核场景，通过该接口使能切K轴。不调用该接口的情况下，默认不切K轴。在GetTiling接口调用前使用。

## 函数原型<a name="section620mcpsimp"></a>

```
void EnableMultiCoreSplitK(bool flag)
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
<tbody><tr id="row106481443135617"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p1487615918216"><a name="p1487615918216"></a><a name="p1487615918216"></a>flag</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p893343112595"><a name="p893343112595"></a><a name="p893343112595"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p0406703116"><a name="p0406703116"></a><a name="p0406703116"></a>是否使能切K轴。</p>
<a name="ul43816319114"></a><a name="ul43816319114"></a><ul id="ul43816319114"><li>true：使能切K轴</li><li>false：不使能切K轴</li></ul>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

-   在算子中使用该接口时，获取C矩阵结果时仅支持输出到Global Memory。
-   在算子中使用该接口时，需在Kernel侧代码中首次将C矩阵分片的结果写入Global Memory之前，先清零Global Memory，随后在获取C矩阵分片的结果时，再开启AtomicAdd累加。如果不预先清零Global Memory，可能会因为累加Global Memory中原始的无效数据而产生精度问题。
-   在算子中使用该接口时，不支持Bias参与矩阵乘计算。

## 调用示例<a name="section046661414719"></a>

```
auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo())
matmul_tiling::MultiCoreMatmulTiling tiling(ascendcPlatform);  

tiling->EnableMultiCoreSplitK(true);  // 使能切K轴
```

