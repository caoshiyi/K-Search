# GetSubBlockIdx<a name="ZH-CN_TOPIC_0000002523343822"></a>

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

获取当前AIV核的ID。分离架构下，一个AI Core由Cube Core（AIC）和Vector Core（AIV）按照一定比例1：N进行组合，其中N个AIV的子核ID分别为0, 1, ..., N-1。

Matmul::GetSubBlockIdx\(\)与基础API接口[AscendC::GetSubBlockIdx\(\)](GetSubBlockIdx(ISASI).md)的区别在于，Matmul::GetSubBlockIdx\(\)用于获取当前AIV核在当前AI Core分组中的ID，而AscendC::GetSubBlockIdx\(\)获取的是AIV核在所有AI Core分组中的逻辑ID。例如，有10组AI Core，AIC与AIV的比例为1:2，共20个AIV核。调用Matmul::GetSubBlockIdx\(\)时，20个AIV的获取结果依次为0, 1, 0, 1, 0, 1, ..., 0, 1。调用AscendC::GetSubBlockIdx\(\)时，20个AIV的获取结果依次为0, 1, 2, 3, 4, 5, ..., 18, 19。

## 函数原型<a name="section620mcpsimp"></a>

```
__aicore__  inline uint8_t GetSubBlockIdx()
```

## 参数说明<a name="section622mcpsimp"></a>

无

## 返回值说明<a name="section640mcpsimp"></a>

当前AIV核的ID。

## 约束说明<a name="section633mcpsimp"></a>

-   该接口仅支持在分离架构下使用，否则返回随机值。
-   在分离架构中，AIV核的ID会在[REGIST\_MATMUL\_OBJ\(\)](REGIST_MATMUL_OBJ.md)接口内部自动初始化和赋值。因此，需要在调用REGIST\_MATMUL\_OBJ\(\)接口之后，再调用本接口，以获取正确的ID。
-   若在算子程序中调用[SetSubBlockIdx\(\)](SetSubBlockIdx.md)后， GetSubBlockIdx\(\)接口将返回由SetSubBlockIdx接口设置的ID值。

## 调用示例<a name="section1665082013318"></a>

```
typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half> aType; 
typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half> bType; 
typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float> cType; 
typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float> biasType; 

AscendC::Matmul<aType, bType, cType, biasType, CFG_NORM> mm;
REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling);
mm.GetSubBlockIdx(); // 获取子核ID
```

