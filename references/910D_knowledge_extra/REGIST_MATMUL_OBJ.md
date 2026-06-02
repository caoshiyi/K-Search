# REGIST\_MATMUL\_OBJ<a name="ZH-CN_TOPIC_0000002554424125"></a>

## 功能说明<a name="section618mcpsimp"></a>

初始化Matmul对象。

## 函数原型<a name="section620mcpsimp"></a>

```
REGIST_MATMUL_OBJ(tpipe, workspace, ...)
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
<tbody><tr id="row106481443135617"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p17933133145916"><a name="p17933133145916"></a><a name="p17933133145916"></a>tpipe</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p893343112595"><a name="p893343112595"></a><a name="p893343112595"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p34832371222"><a name="p34832371222"></a><a name="p34832371222"></a>Tpipe对象。</p>
</td>
</tr>
<tr id="row1491191772211"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p1491121722216"><a name="p1491121722216"></a><a name="p1491121722216"></a>workspace</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p13491141719223"><a name="p13491141719223"></a><a name="p13491141719223"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p2491161714221"><a name="p2491161714221"></a><a name="p2491161714221"></a>系统workspace指针。</p>
</td>
</tr>
<tr id="row1037102214225"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p1538822102212"><a name="p1538822102212"></a><a name="p1538822102212"></a>...</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p53822282215"><a name="p53822282215"></a><a name="p53822282215"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p438182211226"><a name="p438182211226"></a><a name="p438182211226"></a>可变参数，传入Matmul对象和与之对应的Tiling结构，要求Tiling结构的数据类型为<a href="TCubeTiling结构体.md#table1563162142915">TCubeTiling结构</a>。</p>
<p id="p124601322113216"><a name="p124601322113216"></a><a name="p124601322113216"></a>Tiling参数可以通过Host侧<a href="GetTiling.md">GetTiling</a>接口获取，并传递到kernel侧使用。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

-   在[分离模式](基本架构.md#li188191010204418)中，本接口必须在[InitBuffer](InitBuffer.md)接口前调用。
-   在程序中，最多支持定义4个Matmul对象。
-   当代码中只有一个Matmul对象时，本接口可以不传入tiling参数，通过[Init](Init-106.md)接口单独传入tiling参数。
-   当代码中有多个Matmul对象时，必须满足Matmul对象与其tiling参数一一对应，依次传入，具体方式请参考调用示例。

## 调用示例<a name="section15468239131714"></a>

```
Tpipe pipe;
// 推荐：初始化单个matmul对象，传入tiling参数
REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling);
// 推荐：初始化多个matmul对象，传入对应的tiling参数
REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm1, mm1tiling, mm2, mm2tiling, mm3, mm3tiling, mm4, mm4tiling);
// 初始化单个matmul对象，未传入tiling参数。注意，该场景下需要使用Init接口单独传入tiling参数。这种方式将matmul对象的初始化和tiling的设置分离，比如，Tiling可变的场景，可通过这种方式多次对Tiling进行重新设置
REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm);
mm.Init(&tiling);
```

