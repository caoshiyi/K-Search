# Matmul高阶API使能IBShare模板共享A和B矩阵数据<a name="ZH-CN_TOPIC_0000002523289134"></a>

## 案例介绍<a name="section144912211504"></a>

本案例呈现了在融合算子场景中，使用Matmul高阶API进行矩阵乘法计算时，A矩阵和B矩阵同时启用IBShare对性能的提升效果。

该案例的关键优化措施包括：

-   分核逻辑：以Cube核视角分核，Matmul计算结果输出到GM，提供给Vector核进行后续计算。
-   开启IBShare：A矩阵和B矩阵同时开启IBShare。

本案例的算子规格如下：

**表 1**  算子规格

<a name="table568792363119"></a>
<table><thead align="left"><tr id="row1368792319318"><th class="cellrowborder" valign="top" width="25%" id="mcps1.2.5.1.1"><p id="p186887235312"><a name="p186887235312"></a><a name="p186887235312"></a>输入</p>
</th>
<th class="cellrowborder" valign="top" width="25%" id="mcps1.2.5.1.2"><p id="p1268862303114"><a name="p1268862303114"></a><a name="p1268862303114"></a>Shape</p>
</th>
<th class="cellrowborder" valign="top" width="25%" id="mcps1.2.5.1.3"><p id="p1168820237317"><a name="p1168820237317"></a><a name="p1168820237317"></a>Data type</p>
</th>
<th class="cellrowborder" valign="top" width="25%" id="mcps1.2.5.1.4"><p id="p66882235318"><a name="p66882235318"></a><a name="p66882235318"></a>Format</p>
</th>
</tr>
</thead>
<tbody><tr id="row1688142363117"><td class="cellrowborder" valign="top" width="25%" headers="mcps1.2.5.1.1 "><p id="p186887235314"><a name="p186887235314"></a><a name="p186887235314"></a>x</p>
</td>
<td class="cellrowborder" valign="top" width="25%" headers="mcps1.2.5.1.2 "><p id="p7688152310317"><a name="p7688152310317"></a><a name="p7688152310317"></a>128,384</p>
</td>
<td class="cellrowborder" valign="top" width="25%" headers="mcps1.2.5.1.3 "><p id="p3688142393114"><a name="p3688142393114"></a><a name="p3688142393114"></a>float16</p>
</td>
<td class="cellrowborder" valign="top" width="25%" headers="mcps1.2.5.1.4 "><p id="p7688123143119"><a name="p7688123143119"></a><a name="p7688123143119"></a>ND</p>
</td>
</tr>
<tr id="row2688182315313"><td class="cellrowborder" valign="top" width="25%" headers="mcps1.2.5.1.1 "><p id="p268817236313"><a name="p268817236313"></a><a name="p268817236313"></a>y</p>
</td>
<td class="cellrowborder" valign="top" width="25%" headers="mcps1.2.5.1.2 "><p id="p15688182363114"><a name="p15688182363114"></a><a name="p15688182363114"></a>384,256</p>
</td>
<td class="cellrowborder" valign="top" width="25%" headers="mcps1.2.5.1.3 "><p id="p86885235318"><a name="p86885235318"></a><a name="p86885235318"></a>float16</p>
</td>
<td class="cellrowborder" valign="top" width="25%" headers="mcps1.2.5.1.4 "><p id="p20688132373115"><a name="p20688132373115"></a><a name="p20688132373115"></a>ND</p>
</td>
</tr>
</tbody>
</table>

开启IBShare和未开启IBShare的完整样例请参考[MatmulABshare样例](https://gitee.com/ascend/samples/tree/master/operator/ascendc/2_features/13_matmul_api_ibshare)和[MatmulNoABshare样例](https://gitee.com/ascend/samples/blob/master/operator/ascendc/2_features/13_matmul_api_ibshare/MatmulABshareInvocation/matmul_noABshare_custom.cpp)。

## 获取性能数据<a name="section4647105095111"></a>

使用msProf工具获取算子的Profiling的数据，重点分析MTE2，Cube，Scalar的流水情况。

## 分析主要瓶颈点<a name="section371410542511"></a>

**图 1**  优化前Profiling数据<a name="fig516161474220"></a>  
<!-- img2text -->
```
优化前Profiling数据

┌───────────────────────┬───────────┬───────────────┬────────┬──────────┬─────────┬─────────┬─────────────────┬─────────┬─────────┬──────────┬─────────┬──────────┬──────────┬─────────┬─────────┬──────────┬────────────┬────────────┬────────────┬────────────┬───────────┐
│ Op Name               │ Task Type │ Task Duration │ Task W │ aicore_  │ aic_toa │ aic_ma  │ aic_scalar_time │ aic_sca │ aic_mte │ aic_mte2 │ aic_mte │ aic_fixpi │ aic_fixpip│ aic_icach│ aiv_time│ usaiv_total_c│ aiv_vec_tim│ aiv_vec_rat│ aiv_scalar_│ aiv_scalar_l│
├───────────────────────┼───────────┼───────────────┼────────┼──────────┼─────────┼─────────┼─────────────────┼─────────┼─────────┼──────────┼─────────┼──────────┼──────────┼─────────┼─────────┼────────────┼────────────┼────────────┼────────────┼───────────┤
│ matmul_noABshare_custom │ MIX_AIC │ 26.661        │ 2.7    │ 26.3     │ 48589   │ 2.75    │ 25.753          │ 0.981   │ 4.953   │ 0.189    │ 9.854   │ 0.375    │ 15.018   │ 0.572   │ 0.005   │ 20.74      │ 76729      │ 0.042      │ 0.002      │ 4.941 0.238 │
│ matmul_noABshare_custom │ MIX_AIC │ 27.341        │ 2.98   │ 26.9     │ 49818   │ 2.75    │ 26.393          │ 0.98    │ 4.954   │ 0.184    │ 10.048  │ 0.373    │ 15.206   │ 0.565   │ 0.007   │ 21.09      │ 78033      │ 0.042      │ 0.002      │ 4.934 0.234 │
│ matmul_noABshare_custom │ MIX_AIC │ 27.441        │ 2.96   │ 27.1     │ 50123   │ 2.75    │ 26.615          │ 0.982   │ 4.953   │ 0.183    │ 10.152  │ 0.375    │ 15.333   │ 0.566   │ 0.005   │ 21.05      │ 77890      │ 0.042      │ 0.002      │ 4.71  0.224 │
│ matmul_noABshare_custom │ MIX_AIC │ 27.48         │ 2.86   │ 27.1     │ 50142   │ 2.75    │ 26.777          │ 0.988   │ 4.953   │ 0.183    │ 10.17   │ 0.375    │ 15.357   │ 0.567   │ 0.005   │ 21.29      │ 78788      │ 0.042      │ 0.002      │ 4.884 0.229 │
│ matmul_noABshare_custom │ MIX_AIC │ 26.8          │ 2.82   │ 26.4     │ 48870   │ 2.751   │ 26.122          │ 0.989   │ 4.955   │ 0.188    │ 9.935   │ 0.376    │ 15.059   │ 0.57    │ 0.006   │ 20.91      │ 77352      │ 0.042      │ 0.002      │ 5.004 0.239 │
│ matmul_noABshare_custom │ MIX_AIC │ 26.701        │ 2.9    │ 26.3     │ 48653   │ 2.75    │ 25.772          │ 0.98    │ 4.954   │ 0.188    │ 9.923   │ 0.377    │ 15.072   │ 0.573   │ 0.006   │ 20.9       │ 77324      │ 0.042      │ 0.002      │ 4.85  0.232 │
│ matmul_noABshare_custom │ MIX_AIC │ 26.881        │ 2.92   │ 26.5     │ 48939   │ 2.75    │ 25.956          │ 0.981   │ 4.953   │ 0.187    │ 10.169  │ 0.384    │ 15.336   │ 0.58    │ 0.005   │ 20.85      │ 77148      │ 0.042      │ 0.002      │ 4.74  0.227 │
│ matmul_noABshare_custom │ MIX_AIC │ 27.041        │ 2.78   │ 26.6     │ 49225   │ 2.75    │ 26.081          │ 0.98    │ 4.954   │ 0.186    │ 10.25   │ 0.385    │ 15.398   │ 0.579   │ 0.006   │ 21.07      │ 77959      │ 0.042      │ 0.002      │ 4.748 0.225 │
│ matmul_noABshare_custom │ MIX_AIC │ 27.28         │ 2.68   │ 27       │ 49893   │ 2.75    │ 26.488          │ 0.982   │ 4.954   │ 0.184    │ 10.119  │ 0.375    │ 15.292   │ 0.567   │ 0.005   │ 21.15      │ 78248      │ 0.042      │ 0.002      │ 4.745 0.224 │
│ matmul_noABshare_custom │ MIX_AIC │ 27.52         │ 3.02   │ 27.2     │ 50314   │ 2.751   │ 26.672          │ 0.981   │ 4.954   │ 0.182    │ 10.349  │ 0.38     │ 15.529   │ 0.571   │ 0.005   │ 21.42      │ 79243      │ 0.042      │ 0.002      │ 4.774 0.223 │
└───────────────────────┴───────────┴───────────────┴────────┴──────────┴─────────┴─────────┴─────────────────┴─────────┴─────────┴──────────┴─────────┴──────────┴──────────┴─────────┴─────────┴────────────┴────────────┴────────────┴────────────┴───────────┘
```

通过分析以上Profiling数据可以看出，算子执行多次的平均耗时为27.11us，aic\_scalar\_time的平均耗时为26.27us，当前性能瓶颈点为Cube的Scalar流水。

## 设计优化方案<a name="section7611135813517"></a>

A矩阵和B矩阵均未开启IBShare时，数据需要根据K轴、M轴或N轴进行切分计算。这里以K轴切分为例，未开启IBShare之前，算子以AIV Block为视角进行tiling切分，AIV0发起A0\*B0的计算，AIV1发起A1\*B1的计算。

**图 2**  未开启IBShare<a name="fig16885185245110"></a>  
<!-- img2text -->
```text
矩阵A M * K                    矩阵B K * N                         矩阵C M * N

            K                                N                                   N
            │                       ┌────────────────┐                 ┌────────────────┐
            │                       │       B0       │                 │       C0       │
            │                       │                │              M  │                │
M  ┌──────────────────────┐         │                │                 │                │
   │   A0       │   A1     │ K ─────┼────────────────┤                 └────────────────┘
   │            │          │         │       B1       │
   │            │          │         │                │              M  ┌────────────────┐
   └──────────────────────┘         │                │                 │       C1       │
            │                       └────────────────┘                 │                │
            │                                                           │                │
            │                                                           └────────────────┘
                                                                                           N


                         AIV0->Cube : A0 * B0 = C0
                         AIV1->Cube : A1 * B1 = C1
                         result = add(C0,C1)
```

当A矩阵和B矩阵都启用IBShare时，可以一次性加载到L1 Buffer上，省去了切分，分开搬运的过程，同时Cube计算单元完全由AIV0单核驱动，发起一次计算，计算的结果由AIV0和AIV1共享，从而减少Cube响应的次数，减少Scalar计算。

**图 3**  开启IBShare<a name="fig103191116"></a>  

<!-- img2text -->
```
矩阵A M * K                  矩阵B K * N                                  矩阵C M * N

           K                              N                                             N
    M                              K                                        M

┌──────────────────────┐     ┌────────────────┐                     ┌────────────────┐
│                      │     │                │                     │                │
│                      │     │                │                     │                │
│                      │     │                │                     │                │
│                      │     │                │                     │                │
│                      │     │                │                     │                │
└──────────────────────┘     └────────────────┘                     └────────────────┘

AIV0 AIV1->Cube : A * B = C
```

开启IBShare和不开启IBShare的数据交互对比示意图如下：

<!-- img2text -->
```
未开启IBShare

                         ┌──────┐
                         │  L1  │◄──────────────┐
                         └──────┘               │
                            ▲                   │
                            │                   │
┌──────┐                    │                   │        ┌───────────────┐
│ AIC  │────────────────────┘                   ├───────▶│      A0       │◄──┐   ┌──────┐
└──────┘                                        │        ├───────────────┤   │   │ AIV0 │
   │                    Matmul计算               │        │      B0       │◄──┘   └──────┘
   │                                             │        ├───────────────┤
   │                                             │        │ Global        │
   │                                             │        │ Memory        │
   │                                             │        ├───────────────┤   ┌──┐   ┌──────┐
   │                                             └───────▶│      A1       │◄──┤  ├──▶│ AIV1 │
   │                                                      ├───────────────┤   └──┘   └──────┘
   └───────────────────────────────┐                      │      B1       │◄──┐
                                   │                      ├───────────────┤   │
                                   ▼                      │               │   │
                               ┌──────┐                   └───────────────┘   │
                               │ L0C  │────────────────────────────────────────┘
                               └──────┘


AB 矩阵同时开启
IBShare

                         ┌──────┐
                         │  L1  │◄──────────────┐
                         └──────┘               │
                                                │        ┌───────────────┐
┌──────┐                                        ├───────▶│       A       │◄──┐   ┌──────┐
│ AIC  │────────────────────────────────────────┘        ├───────────────┤   │   │ AIV0 │
└──────┘                    Matmul计算                    │ Global        │   │   └──────┘
   │                                                     │ Memory        │   │
   │                                                     ├───────────────┤   │
   │                                                     │       B       │◄──┼───┌──────┐
   │                                                     ├───────────────┤   │   │ AIV1 │
   └───────────────────────────────┐                     │               │   │   └──────┘
                                   │                     └───────────────┘   │
                                   ▼                                          │
                               ┌──────┐                                       │
                               │ L0C  │───────────────────────────────────────┘
                               └──────┘
```

通过设置A和B矩阵MatmulType的IBShare均为true，开启该优化，具体代码如下：

```
constexpr bool isABshare = true;
template <typename aType, typename bType, typename cType> class MatmulABshareKernel {
public:
    __aicore__ inline MatmulABshareKernel(){};
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace,
                                const TCubeTiling &tiling, AscendC::TPipe *pipe);
    __aicore__ inline void Process(AscendC::TPipe *pipe);
    __aicore__ inline void CalcOffset(int32_t blockIdx, const TCubeTiling &tiling, int32_t &offsetA, int32_t &offsetB,
                                      int32_t &offsetC);
    AscendC::Matmul<AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, aType, false, LayoutMode::NONE, isABshare>, 
           AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, bType, false, LayoutMode::NONE, isABshare>,
           AscendC::MatmulType<AscendC::TPosition::VECIN, CubeFormat::ND, cType>>
        matmulObj;
    AscendC::GlobalTensor<aType> aGlobal;
    AscendC::GlobalTensor<bType> bGlobal;
    AscendC::GlobalTensor<cType> cGlobal;
    TCubeTiling tiling;
};
template <typename aType, typename bType, typename cType>
__aicore__ inline void MatmulABshareKernel<aType, bType, cType>::Init(GM_ADDR a, GM_ADDR b, GM_ADDR c, 
                                                                GM_ADDR workspace,const TCubeTiling &tiling, AscendC::TPipe *pipe)
{
    this->tiling = tiling;
    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ aType *>(a), tiling.M * tiling.Ka);
    bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ bType *>(b), tiling.Kb * tiling.N);
    cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ cType *>(c), tiling.M * tiling.N);
    int32_t offsetA, offsetB, offsetC;
    CalcOffset(AscendC::GetBlockIdx(), tiling, offsetA, offsetB, offsetC); // calculate offset
    aGlobal = aGlobal[offsetA];
    bGlobal = bGlobal[offsetB];
    cGlobal = cGlobal[offsetC];
}
template <typename aType, typename bType, typename cType>
__aicore__ inline void
MatmulABshareKernel<aType, bType, cType>::CalcOffset(int32_t blockIdx, const TCubeTiling &tiling,
                                                             int32_t &offsetA, int32_t &offsetB, int32_t &offsetC)
{
    offsetA = 0;
    offsetB = 0;
    offsetC = 0;
}
```

## 验证优化方案性能收益<a name="section8934151165215"></a>

优化后执行多次的平均耗时：22.44us，较优化前有较大提升。

**图 4**  优化后Profiling数据<a name="fig1865995314535"></a>  
<!-- img2text -->
```
图 4  优化后Profiling数据

┌──────────────────────┬───────────┬──────────────┬────────┬───────────┬─────────┬─────────┬─────────────────┬─────────┬─────────┬──────────┬──────────┬──────────┬──────────┬─────────┬──────────┬──────────┬──────────┬──────────┬──────────┬────────────┬────────────┬──────────┬────────────┬────────────┐
│ Op Name              │ Task Type │ Task Duration│ Task W │ aicore_id │ aic_tot │ aic_mac │ aic_scalar_time │ aic_sca │ aic_mte │ aic_mte2 │ aic_mte │ aic_mte2 │ aic_mte │ aic_fixpi│ aic_fixpip│ aic_icach │ aiv_time(us)│ aiv_total_c │ aiv_vec_ti │ aiv_vec_rat│ aiv_scalar_t│ aiv_scalar_ │
├──────────────────────┼───────────┼──────────────┼────────┼───────────┼─────────┼─────────┼─────────────────┼─────────┼─────────┼──────────┼──────────┼──────────┼──────────┼─────────┼──────────┼──────────┼──────────┼──────────┼──────────┼────────────┼────────────┼──────────┼────────────┼────────────┤
│ matmul_ABshare_custom│ MIX_AIC   │ 24.46        │ 2596   │ 24.1      │ 44557   │ 2.486   │ 0.103           │ 19.46   │ 0.808   │ 4.851    │ 0.201    │ 7.862    │ 0.326    │ 9.533   │ 0.396    │ 0.029    │ 21.81     │ 80694     │ 0.022      │ 0.001      │ 3.831      │ 0.176      │
│ matmul_ABshare_custom│ MIX_AIC   │ 22.1         │ 2.8    │ 21.7      │ 40167   │ 2.486   │ 0.115           │ 19.718  │ 0.908   │ 4.856    │ 0.224    │ 8.074    │ 0.372    │ 9.759   │ 0.45     │ 0.026    │ 19.5      │ 72157     │ 0.022      │ 0.001      │ 3.761      │ 0.193      │
│ matmul_ABshare_custom│ MIX_AIC   │ 22.6         │ 2.76   │ 22.3      │ 41213   │ 2.487   │ 0.112           │ 19.883  │ 0.892   │ 4.856    │ 0.218    │ 8.002    │ 0.359    │ 9.719   │ 0.436    │ 0.028    │ 20.08     │ 74306     │ 0.022      │ 0.001      │ 3.912      │ 0.195      │
│ matmul_ABshare_custom│ MIX_AIC   │ 22           │ 2.86   │ 21.7      │ 40052   │ 2.487   │ 0.115           │ 19.363  │ 0.894   │ 4.847    │ 0.224    │ 7.766    │ 0.359    │ 9.394   │ 0.434    │ 0.027    │ 19.6      │ 72531     │ 0.022      │ 0.001      │ 4.002      │ 0.204      │
│ matmul_ABshare_custom│ MIX_AIC   │ 21.66        │ 10.58  │ 21.3      │ 39367   │ 2.487   │ 0.117           │ 19.229  │ 0.904   │ 4.857    │ 0.228    │ 7.713    │ 0.362    │ 9.426   │ 0.443    │ 0.027    │ 19.3      │ 71392     │ 0.022      │ 0.001      │ 3.937      │ 0.204      │
│ matmul_ABshare_custom│ MIX_AIC   │ 21.5         │ 13.88  │ 21.1      │ 39076   │ 2.486   │ 0.118           │ 19.179  │ 0.908   │ 4.867    │ 0.23     │ 7.807    │ 0.37     │ 9.685   │ 0.459    │ 0.026    │ 19.21     │ 71082     │ 0.022      │ 0.001      │ 3.944      │ 0.205      │
│ matmul_ABshare_custom│ MIX_AIC   │ 21.98        │ 12.86  │ 21.6      │ 39950   │ 2.486   │ 0.115           │ 19.512  │ 0.904   │ 4.845    │ 0.224    │ 7.894    │ 0.366    │ 9.542   │ 0.442    │ 0.027    │ 19.48     │ 72059     │ 0.022      │ 0.001      │ 3.824      │ 0.196      │
│ matmul_ABshare_custom│ MIX_AIC   │ 22.28        │ 12.94  │ 21.9      │ 40443   │ 2.486   │ 0.114           │ 19.52   │ 0.893   │ 4.843    │ 0.222    │ 8.095    │ 0.37     │ 9.774   │ 0.447    │ 0.027    │ 19.69     │ 72852     │ 0.022      │ 0.001      │ 3.798      │ 0.193      │
│ matmul_ABshare_custom│ MIX_AIC   │ 23.101       │ 12.7   │ 22.8      │ 42096   │ 2.486   │ 0.109           │ 20.162  │ 0.886   │ 4.852    │ 0.213    │ 8.286    │ 0.364    │ 9.943   │ 0.437    │ 0.027    │ 20.38     │ 75399     │ 0.022      │ 0.001      │ 3.704      │ 0.182      │
│ matmul_ABshare_custom│ MIX_AIC   │ 22.78        │ 22.6   │ 22.4      │ 41463   │ 2.486   │ 0.111           │ 20.383  │ 0.91    │ 4.835    │ 0.216    │ 8.186    │ 0.365    │ 9.862   │ 0.44     │ 0.026    │ 19.97     │ 73878     │ 0.022      │ 0.001      │ 3.929      │ 0.197      │
└──────────────────────┴───────────┴──────────────┴────────┴───────────┴─────────┴─────────┴─────────────────┴─────────┴─────────┴──────────┴──────────┴──────────┴──────────┴─────────┴──────────┴──────────┴──────────┴──────────┴──────────┴────────────┴────────────┴──────────┴────────────┴────────────┘
```

## 总结<a name="section15200958526"></a>

融合算子场景下，Matmul A矩阵和B矩阵同时开启IBShare，以Cube核视角分核，可以有效减少Cube侧的Scalar开销，提升性能。

