# SetSubBlockIdx<a name="ZH-CN_TOPIC_0000002554423537"></a>

## 产品支持情况<a name="section1550532418810"></a>

<a name="table1334714391211"></a>
<table><thead align="left"><tr id="row1334743121213"><th class="cellrowborder" valign="top" width="57.96%" id="mcps1.1.3.1.1"><p id="p834713321216"><a name="p834713321216"></a><a name="p834713321216"></a><span id="ph834783101215"><a name="ph834783101215"></a><a name="ph834783101215"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42.04%" id="mcps1.1.3.1.2"><p id="p2347234127"><a name="p2347234127"></a><a name="p2347234127"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row113472312122"><td class="cellrowborder" valign="top" width="57.96%" headers="mcps1.1.3.1.1 "><p id="p234710320128"><a name="p234710320128"></a><a name="p234710320128"></a><span id="ph103471336127"><a name="ph103471336127"></a><a name="ph103471336127"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42.04%" headers="mcps1.1.3.1.2 "><p id="p4751940181211"><a name="p4751940181211"></a><a name="p4751940181211"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section618mcpsimp"></a>

设置当前AIV核的ID。分离架构下，一个AI Core由Cube Core（AIC）和Vector Core（AIV）按照一定比例1：N进行组合，其中N个AIV核的ID分别为0, 1, ..., N-1。

## 函数原型<a name="section620mcpsimp"></a>

```
__aicore__ inline void SetSubBlockIdx(uint8_t subBlockIdx)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  参数说明

<a name="table6734195885217"></a>
<table><thead align="left"><tr id="row1735115855211"><th class="cellrowborder" valign="top" width="15.191519151915193%" id="mcps1.2.4.1.1"><p id="p197351158205217"><a name="p197351158205217"></a><a name="p197351158205217"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="14.531453145314533%" id="mcps1.2.4.1.2"><p id="p19735058155214"><a name="p19735058155214"></a><a name="p19735058155214"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="70.27702770277028%" id="mcps1.2.4.1.3"><p id="p12735165811523"><a name="p12735165811523"></a><a name="p12735165811523"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row1373511583528"><td class="cellrowborder" valign="top" width="15.191519151915193%" headers="mcps1.2.4.1.1 "><p id="p20467203516917"><a name="p20467203516917"></a><a name="p20467203516917"></a>subBlockIdx</p>
</td>
<td class="cellrowborder" valign="top" width="14.531453145314533%" headers="mcps1.2.4.1.2 "><p id="p273565817523"><a name="p273565817523"></a><a name="p273565817523"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="70.27702770277028%" headers="mcps1.2.4.1.3 "><p id="p1673565819528"><a name="p1673565819528"></a><a name="p1673565819528"></a>当前AIV核的ID。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

-   该接口仅支持在分离架构下使用。
-   在分离架构中，AIV核的ID会在[REGIST\_MATMUL\_OBJ\(\)](REGIST_MATMUL_OBJ.md)接口内部自动初始化和赋值。如果在算子程序中使用了REGIST\_MATMUL\_OBJ\(\)接口，则不建议调用此接口；若未使用REGIST\_MATMUL\_OBJ\(\)接口，则请调用此接口并将子核ID设置为0。

## 调用示例<a name="section1665082013318"></a>

```
typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half> aType; 
typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half> bType; 
typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float> cType; 
typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float> biasType; 

MatmulImpl<aType, bType, cType, biasType, CFG_NORM> mm;
mm.SetSubBlockIdx(0);  // 子核ID设置为0
```

