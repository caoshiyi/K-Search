# SetBatchNum<a name="ZH-CN_TOPIC_0000002523344086"></a>

## 产品支持情况<a name="section1550532418810"></a>

<a name="table1334714391211"></a>
<table><thead align="left"><tr id="row1334743121213"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p834713321216"><a name="p834713321216"></a><a name="p834713321216"></a><span id="ph834783101215"><a name="ph834783101215"></a><a name="ph834783101215"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p2347234127"><a name="p2347234127"></a><a name="p2347234127"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row113472312122"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p234710320128"><a name="p234710320128"></a><a name="p234710320128"></a><span id="ph103471336127"><a name="ph103471336127"></a><a name="ph103471336127"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p4751940181211"><a name="p4751940181211"></a><a name="p4751940181211"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section618mcpsimp"></a>

在不改变Tiling的情况下，重新设置多Batch计算的Batch数。

## 函数原型<a name="section620mcpsimp"></a>

```
__aicore__ inline void SetBatchNum(int32_t batchA, int32_t batchB)
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
<tbody><tr id="row106481443135617"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p17933133145916"><a name="p17933133145916"></a><a name="p17933133145916"></a>batchA</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p893343112595"><a name="p893343112595"></a><a name="p893343112595"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p99335312592"><a name="p99335312592"></a><a name="p99335312592"></a>设置的一次计算的A矩阵Batch数。</p>
</td>
</tr>
<tr id="row142901415103011"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p152909156308"><a name="p152909156308"></a><a name="p152909156308"></a>batchB</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p3290115163016"><a name="p3290115163016"></a><a name="p3290115163016"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p273013218308"><a name="p273013218308"></a><a name="p273013218308"></a>设置的一次计算的B矩阵Batch数。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

-   当使能MixDualMaster（双主模式）场景时，即模板参数[enableMixDualMaster](MatmulConfig.md#p9218181073719)设置为true，不支持使用该接口。
-   本接口仅支持在纯Cube模式（只有矩阵计算）下调用。

## 调用示例<a name="section1665082013318"></a>

```
//  纯cube模式
#define ASCENDC_CUBE_ONLY
typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, AType, false, LayoutMode::NORMAL> aType; 
typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BType, false, LayoutMode::NORMAL> bType;
typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, CType, false, LayoutMode::NORMAL> cType;
typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BiasType> biasType;
AscendC::Matmul<aType, bType, cType, biasType> mm1;
mm1.SetTensorA(gm_a, isTransposeAIn);
mm1.SetTensorB(gm_b, isTransposeBIn);
if(tiling.isBias) {
    mm1.SetBias(gm_bias);
}
mm1.SetBatchNum(batchA, batchB);
// 多batch Matmul计算
mm1.IterateBatch(gm_c, false, 0, false);
```

