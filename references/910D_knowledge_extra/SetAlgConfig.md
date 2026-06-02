# SetAlgConfig<a name="ZH-CN_TOPIC_0000002523303776"></a>

## 功能说明<a name="section618mcpsimp"></a>

设置通信算法。

## 函数原型<a name="section620mcpsimp"></a>

```
uint32_t SetAlgConfig(const std::string &algConfig)
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
<tbody><tr id="row106481443135617"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p167361341213"><a name="p167361341213"></a><a name="p167361341213"></a>algConfig</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p137362417119"><a name="p137362417119"></a><a name="p137362417119"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p915191791319"><a name="p915191791319"></a><a name="p915191791319"></a>通信算法配置。string类型，支持的最大长度为128字节。</p>
<p id="p13299449161211"><a name="p13299449161211"></a><a name="p13299449161211"></a>针对<span id="ph192681844162618"><a name="ph192681844162618"></a><a name="ph192681844162618"></a>Ascend 950PR/Ascend 950DT</span>，该参数为预留字段，配置后不生效，默认仅支持FullMesh算法。FullMesh算法即NPU之间的全连接，任意两个NPU之间可以直接进行数据收发。详细的算法内容可参见<span id="ph16300549101220"><a name="ph16300549101220"></a><a name="ph16300549101220"></a>《HCCL集合通信库用户指南》</span>中的<span id="ph630011499129"><a name="ph630011499129"></a><a name="ph630011499129"></a>相关参考 &gt; 集合通信算法介绍</span>。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

-   0表示设置成功。
-   非0表示设置失败。

## 约束说明<a name="section633mcpsimp"></a>

无

## 调用示例<a name="section1665082013318"></a>

本接口的调用示例请见[调用示例](SetOpType.md#section1665082013318)。

