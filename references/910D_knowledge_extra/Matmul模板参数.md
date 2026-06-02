# Matmul模板参数<a name="ZH-CN_TOPIC_0000002554344223"></a>

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

创建Matmul对象时需要传入：

-   A、B、C、Bias的参数类型信息， 类型信息通过MatmulType来定义，包括：内存逻辑位置、数据格式、数据类型、是否转置、数据排布和是否使能L1复用。
-   MatmulConfig信息（可选），用于配置Matmul模板信息以及相关的配置参数。不配置默认使能Norm模板。
-   MatmulCallBackFunc回调函数信息（可选），用于配置左右矩阵从GM拷贝到A1/B1、计算结果从CO1拷贝到GM的自定义函数。当前支持如下产品型号：

    Ascend 950PR/Ascend 950DT

-   MatmulPolicy信息（可选），用于配置Matmul可拓展模块策略。不配置使能默认模板策略。当前支持如下产品型号：

    Ascend 950PR/Ascend 950DT

## 函数原型<a name="section620mcpsimp"></a>

```
template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE = C_TYPE, const auto& MM_CFG = CFG_NORM, class MM_CB = MatmulCallBackFunc<nullptr, nullptr, nullptr>, MATMUL_POLICY_DEFAULT_OF(MatmulPolicy)>
using Matmul = AscendC::MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>;
```

-   A\_TYPE、B\_TYPE、C\_TYPE类型信息通过[MatmulType](Matmul使用说明.md#table1188045714378)来定义。
-   auto类型的参数MM\_CFG（可选）：
    -   支持MatmulConfig类型：

        Matmul模板信息，具体内容见[MatmulConfig](MatmulConfig.md)。

    -   支持MatmulApiStaticTiling类型：

        MatmulApiStaticTiling参数说明见[表1](#table7939847143412)。

        MatmulApiStaticTiling结构体中包括一组常量化Tiling参数和MatmulConfig结构。这种类型参数的定义方式为，通过调用[MatmulConfig](MatmulConfig.md)章节中介绍的获取模板的接口，指定\(singleM, singleN, singleK, baseM, baseN, baseK\)参数，获取自定义模板；将该模板传入[GetMatmulApiTiling](GetMatmulApiTiling.md)接口，得到常量化的参数。这种常量化的方式将得到MatmulApiStaticTiling结构体中定义的一组常量化参数，可以优化Matmul计算中的Scalar计算。当前支持定义为MatmulApiStaticTiling常量化的Tiling参数的模板有：Norm、IBShare、MDL模板。MxMatmul场景支持定义为MatmulApiStaticTiling常量化的Tiling参数的模板有：Norm、MDL模板。

-   MM\_CB（可选），用于支持不同的搬入搬出需求，实现定制化的搬入搬出功能。具体内容见[MatmulCallBackFunc](MatmulCallBackFunc.md)。
-   MATMUL\_POLICY\_DEFAULT\_OF\(MatmulPolicy\)（可选），用于配置Matmul可拓展模块的策略。当前支持不配置该参数（使能默认模板策略）或者配置1个MatmulPolicy参数。

    MATMUL\_POLICY\_DEFAULT\_OF定义如下，用于简化MATMUL\_POLICY的类型声明。该模板参数的详细使用方式请参考[MatmulPolicy](MatmulPolicy.md)。

    ```
    #define MATMUL_POLICY_DEFAULT_OF(DEFAULT)      \
    template <const auto& = MM_CFG, typename ...>  \
            class MATMUL_POLICY = AscendC::Impl::Detail::DEFAULT
    ```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  MatmulApiStaticTiling常量化Tiling参数说明

<a name="table7939847143412"></a>
<table><thead align="left"><tr id="row1894015475343"><th class="cellrowborder" valign="top" width="16.05%" id="mcps1.2.4.1.1"><p id="p7830192273614"><a name="p7830192273614"></a><a name="p7830192273614"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="34.47%" id="mcps1.2.4.1.2"><p id="p138301122133618"><a name="p138301122133618"></a><a name="p138301122133618"></a>数据类型</p>
</th>
<th class="cellrowborder" valign="top" width="49.480000000000004%" id="mcps1.2.4.1.3"><p id="p783022218365"><a name="p783022218365"></a><a name="p783022218365"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row1794144715341"><td class="cellrowborder" valign="top" width="16.05%" headers="mcps1.2.4.1.1 "><p id="p1789925835613"><a name="p1789925835613"></a><a name="p1789925835613"></a>M, N, Ka, Kb,</p>
<p id="p11899125875617"><a name="p11899125875617"></a><a name="p11899125875617"></a>singleCoreM, singleCoreN, singleCoreK,</p>
<p id="p17899165811566"><a name="p17899165811566"></a><a name="p17899165811566"></a>baseM, baseN, baseK,</p>
<p id="p490012584566"><a name="p490012584566"></a><a name="p490012584566"></a>depthA1, depthB1,</p>
<p id="p139009583566"><a name="p139009583566"></a><a name="p139009583566"></a>stepM， stepN，stepKa，stepKb,</p>
<p id="p2051215216314"><a name="p2051215216314"></a><a name="p2051215216314"></a>isBias,</p>
<p id="p1620315053211"><a name="p1620315053211"></a><a name="p1620315053211"></a>transLength,</p>
<p id="p1820410501321"><a name="p1820410501321"></a><a name="p1820410501321"></a>iterateOrder,</p>
<p id="p107381117133611"><a name="p107381117133611"></a><a name="p107381117133611"></a>dbL0A, dbL0B,</p>
<p id="p4271194073614"><a name="p4271194073614"></a><a name="p4271194073614"></a>dbL0C,</p>
<p id="p102041050153220"><a name="p102041050153220"></a><a name="p102041050153220"></a>shareMode,</p>
<p id="p82041150183214"><a name="p82041150183214"></a><a name="p82041150183214"></a>shareL1Size,</p>
<p id="p551122813336"><a name="p551122813336"></a><a name="p551122813336"></a>shareL0CSize,</p>
<p id="p1851102819332"><a name="p1851102819332"></a><a name="p1851102819332"></a>shareUbSize,</p>
<p id="p1451132883311"><a name="p1451132883311"></a><a name="p1451132883311"></a>batchM,</p>
<p id="p175122873319"><a name="p175122873319"></a><a name="p175122873319"></a>batchN,</p>
<p id="p115222814337"><a name="p115222814337"></a><a name="p115222814337"></a>singleBatchM,</p>
<p id="p152182833318"><a name="p152182833318"></a><a name="p152182833318"></a>singleBatchN,</p>
<p id="p1171123122314"><a name="p1171123122314"></a><a name="p1171123122314"></a>mxTypePara</p>
</td>
<td class="cellrowborder" valign="top" width="34.47%" headers="mcps1.2.4.1.2 "><p id="p1889915586561"><a name="p1889915586561"></a><a name="p1889915586561"></a>int32_t</p>
</td>
<td class="cellrowborder" valign="top" width="49.480000000000004%" headers="mcps1.2.4.1.3 "><p id="p174001310383"><a name="p174001310383"></a><a name="p174001310383"></a>与<a href="TCubeTiling结构体.md">TCubeTiling</a>结构体中各同名参数含义一致。本结构体中的参数是常量化后的常数值。</p>
</td>
</tr>
<tr id="row1550172063919"><td class="cellrowborder" valign="top" width="16.05%" headers="mcps1.2.4.1.1 "><p id="p160684603915"><a name="p160684603915"></a><a name="p160684603915"></a>cfg</p>
</td>
<td class="cellrowborder" valign="top" width="34.47%" headers="mcps1.2.4.1.2 "><p id="p09909561398"><a name="p09909561398"></a><a name="p09909561398"></a><a href="MatmulConfig.md#table1761013213153">MatmulConfig</a></p>
</td>
<td class="cellrowborder" valign="top" width="49.480000000000004%" headers="mcps1.2.4.1.3 "><p id="p36791331513"><a name="p36791331513"></a><a name="p36791331513"></a>Matmul模板的参数配置。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

无

## 调用示例<a name="section1665082013318"></a>

```
// 用户自定义回调函数
void DataCopyOut(const __gm__ void *gm, const LocalTensor<int8_t> &co1Local, const void *dataCopyOutParams, const uint64_t tilingPtr, const uint64_t dataPtr);
void CopyA1(const AscendC::LocalTensor<int8_t> &aMatrix, const __gm__ void *gm, int row, int col, int useM, int useK, const uint64_t tilingPtr, const uint64_t dataPtr);
void CopyB1(const AscendC::LocalTensor<int8_t> &bMatrix, const __gm__ void *gm, int row, int col, int useK, int useN, const uint64_t tilingPtr, const uint64_t dataPtr);

// 定义创建对象时需要传入的A、B、C、Bias参数类型信息
typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half> aType; 
typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half> bType; 
typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float> cType; 
typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float> biasType; 

// 使用MDL模板，创建Matmul实例
AscendC::Matmul<aType, bType, cType, biasType, CFG_MDL> mm1; 

AscendC::MatmulConfig mmConfig{false/*不使能Norm模板*/, true/*使能BasicBlock模板*/, false/*不使能MDL模板*/, 128/*Matmul计算时base块M轴长度*/, 128/*Matmul计算时base块N轴长度*/, 64/*Matmul计算时base块K轴长度*/};
mmConfig.enUnitFlag = false; // 不使能UnitFlag功能
// 使用自定义的mmConfig，创建Matmul实例
AscendC::Matmul<aType, bType, cType, biasType, mmConfig> mm2;

// 使用NORM模板、自定义的mmConfig和自定义的回调函数，创建Matmul实例
AscendC::Matmul<aType, bType, cType, biasType, CFG_NORM, AscendC::MatmulCallBackFunc<DataCopyOut, CopyA1, CopyB1>> mm3;
```

