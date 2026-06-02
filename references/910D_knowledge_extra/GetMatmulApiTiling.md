# GetMatmulApiTiling<a name="ZH-CN_TOPIC_0000002523304612"></a>

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

本接口用于在编译期间获取常量化的Matmul Tiling参数。

Matmul Tiling常量化功能为在编译期期间获取常量化的Matmul Tiling参数并进行算子编译，从而减少Scalar计算开销，提升算子整体性能。具体为，在获取[Matmul模板](Matmul模板参数.md)时，可以确定[MatmulConfig](MatmulConfig.md)的singleCore Shape（[MatmulConfig](MatmulConfig.md)中的singleCoreM/singleCoreN/singleCoreK）和Base Shape（[MatmulConfig](MatmulConfig.md)中的basicM/basicN/basicK）参数，或者只确定Base Shape参数；通过指定获取模板的接口中的singleCore Shape和Base Shape参数，或者只指定Base Shape参数，获取自定义模板；然后调用本接口，得到常量化的Matmul Tiling参数。

当在调用[获取MatmulConfig模板的接口](MatmulConfig.md#li0460173613513)时，只将\(baseM, baseN, baseK\)设置为常数值时，称为部分常量化，此时\(singleCoreM, singleCoreN, singleCoreK\)都保持默认值0，部分常量化场景在Kernel侧使用[REGIST\_MATMUL\_OBJ](REGIST_MATMUL_OBJ.md)初始化Matmul对象时，仍需要使用Tiling；将\(baseM, baseN, baseK, singleCoreM, singleCoreN, singleCoreK\)都设置为常数值时，称为全量常量化，这时可以在[REGIST\_MATMUL\_OBJ](REGIST_MATMUL_OBJ.md)的入参传递Tiling参数的位置，使用空指针替代。

经过上述部分常量化或全部常量化后，将得到带有常量化参数的MatmulConfig模板，然后使用本接口将Tiling参数常量化。本接口的返回值包含常量化的Matmul Tiling参数和MatmulConfig模板。

## 函数原型<a name="section620mcpsimp"></a>

```
template<class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
__aicore__ constexpr MatmulApiStaticTiling GetMatmulApiTiling(const MatmulConfig& mmCFG, int32_t l1Size = Impl::L1_SIZE)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table4835205712588"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001429830437_row118356578583"><th class="cellrowborder" valign="top" width="50%" id="mcps1.2.3.1.1"><p id="zh-cn_topic_0000001429830437_p48354572582"><a name="zh-cn_topic_0000001429830437_p48354572582"></a><a name="zh-cn_topic_0000001429830437_p48354572582"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="50%" id="mcps1.2.3.1.2"><p id="zh-cn_topic_0000001429830437_p583535795817"><a name="zh-cn_topic_0000001429830437_p583535795817"></a><a name="zh-cn_topic_0000001429830437_p583535795817"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001429830437_row1835857145817"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001429830437_p5835457165816"><a name="zh-cn_topic_0000001429830437_p5835457165816"></a><a name="zh-cn_topic_0000001429830437_p5835457165816"></a>A_TYPE</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p9561175153016"><a name="p9561175153016"></a><a name="p9561175153016"></a>A矩阵类型信息，通过<a href="Matmul使用说明.md#table1188045714378">MatmulType</a>来定义。</p>
</td>
</tr>
<tr id="row1648615377"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p1212015191874"><a name="p1212015191874"></a><a name="p1212015191874"></a>B_TYPE</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p1912061914715"><a name="p1912061914715"></a><a name="p1912061914715"></a>B矩阵类型信息，通过<a href="Matmul使用说明.md#table1188045714378">MatmulType</a>来定义。</p>
</td>
</tr>
<tr id="row068983542919"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p12689103517298"><a name="p12689103517298"></a><a name="p12689103517298"></a>C_TYPE</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p27722268302"><a name="p27722268302"></a><a name="p27722268302"></a>C矩阵类型信息，通过<a href="Matmul使用说明.md#table1188045714378">MatmulType</a>来定义。</p>
</td>
</tr>
<tr id="row227517394296"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p15275143942912"><a name="p15275143942912"></a><a name="p15275143942912"></a>BIAS_TYPE</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p13660122733017"><a name="p13660122733017"></a><a name="p13660122733017"></a>BIAS矩阵类型信息，通过<a href="Matmul使用说明.md#table1188045714378">MatmulType</a>来定义。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  参数说明

<a name="table9646134355611"></a>
<table><thead align="left"><tr id="row964714433565"><th class="cellrowborder" valign="top" width="14.99%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="12.02%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="72.99%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row10312181416266"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p6307141482619"><a name="p6307141482619"></a><a name="p6307141482619"></a>mmCFG</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p1783442222713"><a name="p1783442222713"></a><a name="p1783442222713"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p63071114122618"><a name="p63071114122618"></a><a name="p63071114122618"></a>获取的<a href="MatmulConfig.md#table1761013213153">MatmulConfig</a>模板。</p>
<p id="p4805114144610"><a name="p4805114144610"></a><a name="p4805114144610"></a>对于<span id="ph280519415461"><a name="ph280519415461"></a><a name="ph280519415461"></a>Ascend 950PR/Ascend 950DT</span>，支持常量化的为全部模板：Norm, IBShare, MDL模板。</p>
</td>
</tr>
<tr id="row431281415261"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p123074143264"><a name="p123074143264"></a><a name="p123074143264"></a>l1Size</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p12307814132615"><a name="p12307814132615"></a><a name="p12307814132615"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p831221402620"><a name="p831221402620"></a><a name="p831221402620"></a>可用的L1大小，默认值L1_SIZE。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

[MatmulApiStaticTiling](Matmul模板参数.md#table7939847143412)

## 约束说明<a name="section633mcpsimp"></a>

-   入参mmCFG，在调用获取MatmulConfig模板的接口获取时，需要使用常数值指定\(baseM, baseN, baseK\)或者指定\(baseM, baseN, baseK, singleCoreM, singleCoreN, singleCoreK\)，并且指定的参数值需要和Tiling计算的值保持一致。
-   Batch Matmul场景支持全量常量化，但不支持使用空指针替代REGIST\_MATMUL\_OBJ的入参Tiling。

## 调用示例<a name="section98751211152118"></a>

完整算子样例请参考[Matmul Tiling常量化的算子样例](https://gitcode.com/cann/asc-devkit/tree/master/examples/03_libraries/01_matrix/matmul_constant)。

```
//定义Matmul对象
typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half> aType; 
typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half> bType; 
typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float> cType; 
typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float> biasType; 
// 这里CFG使用GetNormalConfig接口获取，并指定已知的singleshape信息和baseM,baseN,baseK，指定的数值跟运行时tiling保持一致
constexpr auto staticTiling = GetMatmulApiTiling<aType, bType, cType, biasType>(CFG, 524288); // 该示例L1 Buffer可用大小为512KB
AscendC::Matmul<aType, bType, cType, biasType, staticTiling > mm; 
```

