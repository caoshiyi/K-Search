# SetMatmulConfigParams<a name="ZH-CN_TOPIC_0000002523344762"></a>

## 功能说明<a name="section618mcpsimp"></a>

在计算Tiling时，用于自定义设置[表1](#table9646134355611)中的MatmulConfig参数。本接口中配置的参数对应的功能在Tiling与Kernel中需要保持一致，所以本接口中的参数取值，需要与Kernel侧对应的MatmulConfig参数值保持一致，详细MatmulConfig参数请见[表2](MatmulConfig.md#table1761013213153)。

## 函数原型<a name="section1892893914372"></a>

```
void SetMatmulConfigParams(int32_t mmConfigTypeIn = 1, bool enableL1CacheUBIn = false, ScheduleType scheduleTypeIn = ScheduleType::INNER_PRODUCT, MatrixTraverse traverseIn = MatrixTraverse::NOSET, bool enVecND2NZIn = false)
```

```
void SetMatmulConfigParams(const MatmulConfigParams& configParams)
```

## 参数说明<a name="section61221956113713"></a>

**表 1**  参数说明

<a name="table9646134355611"></a>
<table><thead align="left"><tr id="row964714433565"><th class="cellrowborder" valign="top" width="17.82%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="9.19%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="72.99%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row106481443135617"><td class="cellrowborder" valign="top" width="17.82%" headers="mcps1.2.4.1.1 "><p id="p1487615918216"><a name="p1487615918216"></a><a name="p1487615918216"></a>mmConfigTypeIn</p>
</td>
<td class="cellrowborder" valign="top" width="9.19%" headers="mcps1.2.4.1.2 "><p id="p893343112595"><a name="p893343112595"></a><a name="p893343112595"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p1280216522410"><a name="p1280216522410"></a><a name="p1280216522410"></a>设置Matmul的模板类型，需要与Matmul对象创建的模板一致，当前只支持配置为0或1。</p>
<a name="ul1516211562241"></a><a name="ul1516211562241"></a><ul id="ul1516211562241"><li>0：代表Norm模板</li><li>1：代表MDL模板，默认值为1</li></ul>
</td>
</tr>
<tr id="row36481043185619"><td class="cellrowborder" valign="top" width="17.82%" headers="mcps1.2.4.1.1 "><p id="p293333110598"><a name="p293333110598"></a><a name="p293333110598"></a>enableL1CacheUBIn</p>
</td>
<td class="cellrowborder" valign="top" width="9.19%" headers="mcps1.2.4.1.2 "><p id="p7933731135920"><a name="p7933731135920"></a><a name="p7933731135920"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p7775104382514"><a name="p7775104382514"></a><a name="p7775104382514"></a>配置是否使能L1缓存UB计算块；参考使能场景：MTE3和MTE2流水串行较多的场景。</p>
<a name="ul1675991252812"></a><a name="ul1675991252812"></a><ul id="ul1675991252812"><li>false：不使能L1缓存UB计算块，默认值为false</li><li>true：使能L1缓存UB计算块</li></ul>
</td>
</tr>
<tr id="row1238115883"><td class="cellrowborder" valign="top" width="17.82%" headers="mcps1.2.4.1.1 "><p id="p12831013171011"><a name="p12831013171011"></a><a name="p12831013171011"></a>scheduleTypeIn</p>
</td>
<td class="cellrowborder" valign="top" width="9.19%" headers="mcps1.2.4.1.2 "><p id="p11369102020104"><a name="p11369102020104"></a><a name="p11369102020104"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p162831613101015"><a name="p162831613101015"></a><a name="p162831613101015"></a>配置Matmul数据搬运模式。参数取值如下：</p>
<a name="ul9283191317107"></a><a name="ul9283191317107"></a><ul id="ul9283191317107"><li>ScheduleType::INNER_PRODUCT：默认模式，在K方向上做MTE1的循环搬运</li><li>ScheduleType::OUTER_PRODUCT：在M或N方向上做MTE1的循环搬运</li><li>ScheduleType::N_BUFFER_33：<a href="MatmulPolicy.md#li194081238103913">NBuffer33</a>模板的数据搬运模式，MTE2每次搬运A矩阵的1x3个基本块，直至A矩阵所有3x3个基本块全载在L1 Buffer中</li></ul>
</td>
</tr>
<tr id="row072884131011"><td class="cellrowborder" valign="top" width="17.82%" headers="mcps1.2.4.1.1 "><p id="p16283151311106"><a name="p16283151311106"></a><a name="p16283151311106"></a>traverseIn</p>
</td>
<td class="cellrowborder" valign="top" width="9.19%" headers="mcps1.2.4.1.2 "><p id="p113691020121018"><a name="p113691020121018"></a><a name="p113691020121018"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p528316139109"><a name="p528316139109"></a><a name="p528316139109"></a>Matmul做矩阵运算的循环迭代顺序，即一次迭代计算出[baseM, baseN]大小的C矩阵分片后，自动偏移到下一次迭代输出的C矩阵位置的偏移顺序。参数取值如下：</p>
<p id="p19628162911110"><a name="p19628162911110"></a><a name="p19628162911110"></a>NOSET：0，当前无效。</p>
<p id="p1166711451116"><a name="p1166711451116"></a><a name="p1166711451116"></a>FIRSTM：先往M轴方向偏移再往N轴方向偏移。</p>
<p id="p1740819561117"><a name="p1740819561117"></a><a name="p1740819561117"></a>FIRSTN：先往N轴方向偏移再往M轴方向偏移。</p>
</td>
</tr>
<tr id="row1149633818115"><td class="cellrowborder" valign="top" width="17.82%" headers="mcps1.2.4.1.1 "><p id="p059945634917"><a name="p059945634917"></a><a name="p059945634917"></a>enVecND2NZIn</p>
</td>
<td class="cellrowborder" valign="top" width="9.19%" headers="mcps1.2.4.1.2 "><p id="p759985634919"><a name="p759985634919"></a><a name="p759985634919"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p1781153815011"><a name="p1781153815011"></a><a name="p1781153815011"></a>是否使能ND2NZ。</p>
</td>
</tr>
<tr id="row137121245182714"><td class="cellrowborder" valign="top" width="17.82%" headers="mcps1.2.4.1.1 "><p id="p1971217453272"><a name="p1971217453272"></a><a name="p1971217453272"></a>configParams</p>
</td>
<td class="cellrowborder" valign="top" width="9.19%" headers="mcps1.2.4.1.2 "><p id="p971294513279"><a name="p971294513279"></a><a name="p971294513279"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p17376814155615"><a name="p17376814155615"></a><a name="p17376814155615"></a>config相关参数，类型为MatmulConfigParams，结构体具体定义如下方代码所示。其中的参数说明请参考<a href="#table15780447181917">表2</a>。</p>
</td>
</tr>
</tbody>
</table>

```
struct MatmulConfigParams
{
    int32_t mmConfigType;
    bool enableL1CacheUB;
    ScheduleType scheduleType;
    MatrixTraverse traverse;
    bool enVecND2NZ;
    MatmulConfigParams(int32_t mmConfigTypeIn = 1, bool enableL1CacheUBIn = false,
        ScheduleType scheduleTypeIn = ScheduleType::INNER_PRODUCT, MatrixTraverse traverseIn = MatrixTraverse::NOSET,
        bool enVecND2NZIn = false) {
        mmConfigType = mmConfigTypeIn;
        enableL1CacheUB = enableL1CacheUBIn;
        scheduleType = scheduleTypeIn;
        traverse = traverseIn;
        enVecND2NZ = enVecND2NZIn;
    }
};
```

**表 2**  MatmulConfigParams结构体内参数说明

<a name="table15780447181917"></a>
<table><thead align="left"><tr id="row0780947111915"><th class="cellrowborder" valign="top" width="16.61%" id="mcps1.2.3.1.1"><p id="p1780124771913"><a name="p1780124771913"></a><a name="p1780124771913"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="83.39%" id="mcps1.2.3.1.2"><p id="p1578014718198"><a name="p1578014718198"></a><a name="p1578014718198"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row10780647151919"><td class="cellrowborder" valign="top" width="16.61%" headers="mcps1.2.3.1.1 "><p id="p6340835122118"><a name="p6340835122118"></a><a name="p6340835122118"></a>mmConfigType</p>
</td>
<td class="cellrowborder" valign="top" width="83.39%" headers="mcps1.2.3.1.2 "><p id="p1724564423319"><a name="p1724564423319"></a><a name="p1724564423319"></a>设置Matmul的模板类型，需要与Matmul对象创建的模板一致，当前只支持配置为0或1。</p>
<a name="ul5245174463318"></a><a name="ul5245174463318"></a><ul id="ul5245174463318"><li>0：代表Norm模板</li><li>1：代表MDL模板，默认值为1</li></ul>
</td>
</tr>
<tr id="row6780947191919"><td class="cellrowborder" valign="top" width="16.61%" headers="mcps1.2.3.1.1 "><p id="p1934033512213"><a name="p1934033512213"></a><a name="p1934033512213"></a>enableL1CacheUB</p>
</td>
<td class="cellrowborder" valign="top" width="83.39%" headers="mcps1.2.3.1.2 "><p id="p1836962183616"><a name="p1836962183616"></a><a name="p1836962183616"></a>配置是否使能L1缓存UB计算块；参考使能场景：MTE3和MTE2流水串行较多的场景。</p>
<a name="ul736962123618"></a><a name="ul736962123618"></a><ul id="ul736962123618"><li>false：不使能L1缓存UB计算块，默认值为false</li><li>true：使能L1缓存UB计算块</li></ul>
</td>
</tr>
<tr id="row1078074711194"><td class="cellrowborder" valign="top" width="16.61%" headers="mcps1.2.3.1.1 "><p id="p334033518217"><a name="p334033518217"></a><a name="p334033518217"></a>scheduleType</p>
</td>
<td class="cellrowborder" valign="top" width="83.39%" headers="mcps1.2.3.1.2 "><p id="p1993255310368"><a name="p1993255310368"></a><a name="p1993255310368"></a>配置Matmul数据搬运模式。参数取值如下：</p>
<a name="ul1893214539369"></a><a name="ul1893214539369"></a><ul id="ul1893214539369"><li>ScheduleType::INNER_PRODUCT：默认模式，在K方向上做MTE1的循环搬运</li><li>ScheduleType::OUTER_PRODUCT：在M或N方向上做MTE1的循环搬运</li><li>ScheduleType::N_BUFFER_33：<a href="MatmulPolicy.md#li194081238103913">NBuffer33</a>模板的数据搬运模式，MTE2每次搬运A矩阵的1x3个基本块，直至A矩阵所有3x3个基本块全载在L1 Buffer中</li></ul>
</td>
</tr>
<tr id="row1761285762117"><td class="cellrowborder" valign="top" width="16.61%" headers="mcps1.2.3.1.1 "><p id="p6368183273219"><a name="p6368183273219"></a><a name="p6368183273219"></a>traverse</p>
</td>
<td class="cellrowborder" valign="top" width="83.39%" headers="mcps1.2.3.1.2 "><p id="p26801772455"><a name="p26801772455"></a><a name="p26801772455"></a>Matmul做矩阵运算的循环迭代顺序，即一次迭代计算出[baseM, baseN]大小的C矩阵分片后，自动偏移到下一次迭代输出的C矩阵位置的偏移顺序。参数取值如下：</p>
<p id="p17190174361212"><a name="p17190174361212"></a><a name="p17190174361212"></a>NOSET：0，当前无效。</p>
<p id="p062802111316"><a name="p062802111316"></a><a name="p062802111316"></a>FIRSTM：先往M轴方向偏移再往N轴方向偏移。</p>
<p id="p410311311312"><a name="p410311311312"></a><a name="p410311311312"></a>FIRSTN：先往N轴方向偏移再往M轴方向偏移。</p>
</td>
</tr>
<tr id="row1158214231986"><td class="cellrowborder" valign="top" width="16.61%" headers="mcps1.2.3.1.1 "><p id="p48862519810"><a name="p48862519810"></a><a name="p48862519810"></a>enVecND2NZ</p>
</td>
<td class="cellrowborder" valign="top" width="83.39%" headers="mcps1.2.3.1.2 "><p id="p1689192520817"><a name="p1689192520817"></a><a name="p1689192520817"></a>是否使能ND2NZ</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section26151511103814"></a>

无

## 约束说明<a name="section951491643810"></a>

-   本接口必须在[GetTiling](GetTiling.md)接口前调用。
-   若Matmul对象使用NBuffer33模板策略，即MatmulPolicy为[NBuffer33MatmulPolicy](MatmulPolicy.md#li194081238103913)，则在调用[GetTiling](GetTiling.md)接口生成Tiling参数前，必须通过本接口将scheduleTypeIn参数设置为ScheduleType::N\_BUFFER\_33，以启用NBuffer33模板策略的Tiling生成逻辑。

## 调用示例<a name="section1665082013318"></a>

```
auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
matmul_tiling::MatmulApiTiling tiling(ascendcPlatform); 
tiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16); 
tiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);   
tiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);   
tiling.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);   
tiling.SetShape(1024, 1024, 1024);   
tiling.SetOrgShape(1024, 1024, 1024);  
tiling.SetBias(true);   
tiling.SetBufferSpace(-1, -1, -1);
tiling.SetMatmulConfigParams(0);  // 额外设置
// matmul_tiling::MatmulConfigParams configParams = {1, false, matmul_tiling::ScheduleType::OUTER_PRODUCT, matmul_tiling::MatrixTraverse::FIRSTM};
// tiling.SetMatmulConfigParams(configParams);
optiling::TCubeTiling tilingData;   
int ret = tiling.GetTiling(tilingData);
```

