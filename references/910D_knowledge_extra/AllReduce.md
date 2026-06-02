# AllReduce<a name="ZH-CN_TOPIC_0000002554424815"></a>

## 产品支持情况<a name="section1586581915393"></a>

<a name="table169596713360"></a>
<table><thead align="left"><tr id="row129590715369"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p17959971362"><a name="p17959971362"></a><a name="p17959971362"></a><span id="ph895914718367"><a name="ph895914718367"></a><a name="ph895914718367"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p89594763612"><a name="p89594763612"></a><a name="p89594763612"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row18959673369"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p1595910763613"><a name="p1595910763613"></a><a name="p1595910763613"></a><span id="ph1595918753613"><a name="ph1595918753613"></a><a name="ph1595918753613"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p1695957133611"><a name="p1695957133611"></a><a name="p1695957133611"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section618mcpsimp"></a>

集合通信算子AllReduce的任务下发接口，返回该任务的标识handleId给用户。AllReduce功能为：将通信域内所有节点的同名张量进行reduce操作后，再把结果发送到所有节点的输出buffer。

<!-- img2text -->
```text
                rank0      rank1      rank2      rank3
              ┌────────┬────────┬────────┬────────┐
              │        │        │        │        │
              │  in0   │  in1   │  in2   │  in3   │
              │        │        │        │        │
              └────────┴────────┴────────┴────────┘
                           AllReduce
              ───────────────────────→
                rank0      rank1      rank2      rank3
              ┌────────┬────────┬────────┬────────┐
              │        │        │        │        │
              │  out   │  out   │  out   │  out   │
              │        │        │        │        │
              └────────┴────────┴────────┴────────┘

若操作类型为sum，则out[i]=sum(inX[i])
```

## 函数原型<a name="section620mcpsimp"></a>

```
template <bool commit = false>
__aicore__ inline HcclHandle AllReduce(GM_ADDR sendBuf, GM_ADDR recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op, uint8_t repeat = 1)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="zh-cn_topic_0235751031_table33761356"></a>
<table><thead align="left"><tr id="zh-cn_topic_0235751031_row27598891"><th class="cellrowborder" valign="top" width="17.77%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="11.799999999999999%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="70.43%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row42461942101815"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p163481714145518"><a name="p163481714145518"></a><a name="p163481714145518"></a>commit</p>
</td>
<td class="cellrowborder" valign="top" width="11.799999999999999%" headers="mcps1.2.4.1.2 "><p id="p33487148556"><a name="p33487148556"></a><a name="p33487148556"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="70.43%" headers="mcps1.2.4.1.3 "><p id="p186182538493"><a name="p186182538493"></a><a name="p186182538493"></a>bool类型。参数取值如下：</p>
<a name="ul77246714401"></a><a name="ul77246714401"></a><ul id="ul77246714401"><li>true：在调用Prepare接口时，Commit同步通知服务端可以执行该通信任务。</li><li>false：在调用Prepare接口时，不通知服务端执行该通信任务。</li></ul>
</td>
</tr>
</tbody>
</table>

**表 2**  接口参数说明

<a name="table180119381514"></a>
<table><thead align="left"><tr id="row148011835158"><th class="cellrowborder" valign="top" width="17.77%" id="mcps1.2.4.1.1"><p id="p1280114381517"><a name="p1280114381517"></a><a name="p1280114381517"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="11.799999999999999%" id="mcps1.2.4.1.2"><p id="p380111321517"><a name="p380111321517"></a><a name="p380111321517"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="70.43%" id="mcps1.2.4.1.3"><p id="p28014351520"><a name="p28014351520"></a><a name="p28014351520"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row17761811191614"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p167771011181619"><a name="p167771011181619"></a><a name="p167771011181619"></a>sendBuf</p>
</td>
<td class="cellrowborder" valign="top" width="11.799999999999999%" headers="mcps1.2.4.1.2 "><p id="p377721181614"><a name="p377721181614"></a><a name="p377721181614"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="70.43%" headers="mcps1.2.4.1.3 "><p id="p1477711161611"><a name="p1477711161611"></a><a name="p1477711161611"></a>源数据buffer地址。</p>
</td>
</tr>
<tr id="row11249349125111"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p12924053195111"><a name="p12924053195111"></a><a name="p12924053195111"></a>recvBuf</p>
</td>
<td class="cellrowborder" valign="top" width="11.799999999999999%" headers="mcps1.2.4.1.2 "><p id="p2924553115118"><a name="p2924553115118"></a><a name="p2924553115118"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="70.43%" headers="mcps1.2.4.1.3 "><p id="p1192485395112"><a name="p1192485395112"></a><a name="p1192485395112"></a>目的数据buffer地址，集合通信结果输出到此buffer中。</p>
</td>
</tr>
<tr id="row17902814191610"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p2090218149169"><a name="p2090218149169"></a><a name="p2090218149169"></a>count</p>
</td>
<td class="cellrowborder" valign="top" width="11.799999999999999%" headers="mcps1.2.4.1.2 "><p id="p0902714111616"><a name="p0902714111616"></a><a name="p0902714111616"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="70.43%" headers="mcps1.2.4.1.3 "><p id="p5902191421615"><a name="p5902191421615"></a><a name="p5902191421615"></a>参与AllReduce操作的数据个数，比如只有一个int32数据参与，则count=1。</p>
</td>
</tr>
<tr id="row48213177160"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p158219172167"><a name="p158219172167"></a><a name="p158219172167"></a>dataType</p>
</td>
<td class="cellrowborder" valign="top" width="11.799999999999999%" headers="mcps1.2.4.1.2 "><p id="p118152714175"><a name="p118152714175"></a><a name="p118152714175"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="70.43%" headers="mcps1.2.4.1.3 "><p id="p98211178163"><a name="p98211178163"></a><a name="p98211178163"></a>AllReduce操作的数据类型，目前支持float、half（即float16）、int8_t、int16_t、int32_t、bfloat16_t数据类型，即支持取值为HCCL_DATA_TYPE_FP32、HCCL_DATA_TYPE_FP16、HCCL_DATA_TYPE_INT8、HCCL_DATA_TYPE_INT16、HCCL_DATA_TYPE_INT32、HCCL_DATA_TYPE_BFP16。HcclDataType数据类型的介绍请参考<a href="HCCL使用说明.md#table116710585514">表1</a>。</p>
</td>
</tr>
<tr id="row1481814577161"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p3818657171610"><a name="p3818657171610"></a><a name="p3818657171610"></a>op</p>
</td>
<td class="cellrowborder" valign="top" width="11.799999999999999%" headers="mcps1.2.4.1.2 "><p id="p11490527181714"><a name="p11490527181714"></a><a name="p11490527181714"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="70.43%" headers="mcps1.2.4.1.3 "><p id="p208199571169"><a name="p208199571169"></a><a name="p208199571169"></a>Reduce的操作类型，目前支持sum、max、min操作类型，即支持取值为HCCL_REDUCE_SUM、HCCL_REDUCE_MAX、HCCL_REDUCE_MIN。HcclReduceOp数据类型的介绍请参考<a href="HCCL使用说明.md#table2469980529">表2</a>。</p>
</td>
</tr>
<tr id="row165671519176"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p17656215111710"><a name="p17656215111710"></a><a name="p17656215111710"></a>repeat</p>
</td>
<td class="cellrowborder" valign="top" width="11.799999999999999%" headers="mcps1.2.4.1.2 "><p id="p38942811174"><a name="p38942811174"></a><a name="p38942811174"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="70.43%" headers="mcps1.2.4.1.3 "><p id="p15631125672811"><a name="p15631125672811"></a><a name="p15631125672811"></a>一次下发的AllReduce通信任务个数。repeat取值≥1，默认值为1。当repeat&gt;1时，每个AllReduce任务的sendBuf和recvBuf地址由服务端自动算出，计算公式如下：</p>
<p id="p91522315916"><a name="p91522315916"></a><a name="p91522315916"></a>sendBuf[i] = sendBuf + count* sizeof(datatype) * i, i∈[0, repeat)</p>
<p id="p19310955081"><a name="p19310955081"></a><a name="p19310955081"></a>recvBuf[i] = recvBuf + count* sizeof(datatype) * i, i∈[0, repeat)</p>
<p id="p777292611013"><a name="p777292611013"></a><a name="p777292611013"></a>注意：当设置repeat&gt;1时，须与count参数配合使用，规划通信数据地址。</p>
</td>
</tr>
</tbody>
</table>

**图 1**  AllReduce三轮切分通信示例<a name="fig5638165920312"></a>  
<!-- img2text -->
```text
                     rank0                                             rank0 recvBuf
              ┌────────────────┐                                   ┌──────────────────────────┐
count ────────┤      0-0       │──────────────────┐        ┌──────▶│ 0-0 + 1-0 + 2-0 + 3-0  │
              ├────────────────┤                  │        │       ├──────────────────────────┤
              │      0-1       │                  │        │       │ 0-1 + 1-1 + 2-1 + 3-1  │
              ├ - - - - - - - ─┤                  │        │       ├──────────────────────────┤
              │      0-2       │                  │        │       │ 0-2 + 1-2 + 2-2 + 3-2  │
              └────────────────┘                  │        │       └──────────────────────────┘
                                                  │        │
                     rank1                        │        │              rank1 recvBuf
              ┌────────────────┐                  │        │       ┌──────────────────────────┐
              │      1-0       │──────────────┐   │        │   ┌──▶│ 0-0 + 1-0 + 2-0 + 3-0  │
              ├────────────────┤              │   │        │   │   ├──────────────────────────┤
              │      1-1       │              │   │        │   │   │ 0-1 + 1-1 + 2-1 + 3-1  │
              ├ - - - - - - - ─┤              │   │        │   │   ├──────────────────────────┤
              │      1-2       │              │   │        │   │   │ 0-2 + 1-2 + 2-2 + 3-2  │
              └────────────────┘              │   │        │   │   └──────────────────────────┘
                                              │   │        │   │
                     rank2                    │   │        │   │              rank2 recvBuf
              ┌────────────────┐              │   │        │   │   ┌──────────────────────────┐
              │      2-0       │──────────────┼───┼────────┼───┼──▶│ 0-0 + 1-0 + 2-0 + 3-0  │
              ├────────────────┤              │   │        │   │   ├──────────────────────────┤
              │      2-1       │              │   │        │   │   │ 0-1 + 1-1 + 2-1 + 3-1  │
              ├ - - - - - - - ─┤              │   │        │   │   ├──────────────────────────┤
              │      2-2       │              │   │        │   │   │ 0-2 + 1-2 + 2-2 + 3-2  │
              └────────────────┘              │   │        │   │   └──────────────────────────┘
                                              │   │        │   │
                     rank3                    │   │        │   │              rank3 recvBuf
              ┌────────────────┐              │   │        │   │   ┌──────────────────────────┐
              │      3-0       │──────────────┘   │        │   └──▶│ 0-0 + 1-0 + 2-0 + 3-0  │
              ├────────────────┤                  │        │       ├──────────────────────────┤
              │      3-1       │──────────────────┼────────┼──────▶│ 0-1 + 1-1 + 2-1 + 3-1  │
              ├ - - - - - - - ─┤                  │        │       ├──────────────────────────┤
              │      3-2       │──────────────────┘        └──────▶│ 0-2 + 1-2 + 2-2 + 3-2  │
              └────────────────┘                                   └──────────────────────────┘

                                         ┌────────────┐
                                         │ AllReduce  │
                                         │   结果     │
                                         └────────────┘
```

## 返回值说明<a name="section640mcpsimp"></a>

返回该任务的标识handleId，handleId大于等于0。调用失败时，返回 -1。

## 约束说明<a name="section633mcpsimp"></a>

-   调用本接口前确保已调用过[InitV2](InitV2.md)和[SetCcTilingV2](SetCcTilingV2.md)接口。
-   若HCCL对象的[config模板参数](HCCL模板参数.md#table884518212555)未指定下发通信任务的核，该接口只能在AIC核或者AIV核两者之一上调用。若HCCL对象的[config模板参数](HCCL模板参数.md#table884518212555)中指定了下发通信任务的核，则该接口可以在AIC核和AIV核上同时调用，接口内部会根据指定的核的类型，只在AIC核、AIV核二者之一下发该通信任务。
-   对于Ascend 950PR/Ascend 950DT，一个通信域内，所有Prepare接口的总调用次数不能超过63。
-   对于Ascend 950PR/Ascend 950DT，通信服务端为CCU时，单次最大通信数据量不能超过256M。

## 调用示例<a name="section1665082013318"></a>

-   非多轮切分场景

    如下图所示，4张卡上均有count=300个float16数据，每张卡从xGM内存中获取到本卡数据，各卡的数据进行reduce sum计算后，将结果输出到各卡的yGM。

    **图 2**  非多轮切分场景下4卡AllReduce通信<a name="fig101532392516"></a>  
    
    <!-- img2text -->
```text
                    rank0 xGM                                   rank0 yGM
              ┌─────────────────┐                        ┌─────────────────┐
count=300     │       0-0       │                        │ 0-0 + 1-0 + 2-0 │
     ├────────│                 │                        │      + 3-0      │
              └─────────────────┘                        └─────────────────┘

                    rank1 xGM                                   rank1 yGM
              ┌─────────────────┐                        ┌─────────────────┐
              │       1-0       │                        │ 0-0 + 1-0 + 2-0 │
              │                 │────── AllReduce ───→   │      + 3-0      │
              └─────────────────┘                        └─────────────────┘

                    rank2 xGM                                   rank2 yGM
              ┌─────────────────┐                        ┌─────────────────┐
              │       2-0       │                        │ 0-0 + 1-0 + 2-0 │
              │                 │                        │      + 3-0      │
              └─────────────────┘                        └─────────────────┘

                    rank3 xGM                                   rank3 yGM
              ┌─────────────────┐                        ┌─────────────────┐
              │       3-0       │                        │ 0-0 + 1-0 + 2-0 │
              │                 │                        │      + 3-0      │
              └─────────────────┘                        └─────────────────┘
```

    ```
    extern "C" __global__ __aicore__ void all_reduce_custom(GM_ADDR xGM, GM_ADDR yGM, GM_ADDR workspaceGM, GM_ADDR tilingGM) {
        auto sendBuf = xGM;  // xGM为AllReduce的输入GM地址
        auto recvBuf = yGM;  // yGM为AllReduce的输出GM地址
        uint64_t sendCount = 300;  // 每张卡上均有300个float16的数据
        HcclReduceOp reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
        REGISTER_TILING_DEFAULT(AllReduceCustomTilingData); //AllReduceCustomTilingData为对应算子头文件定义的结构体
        GET_TILING_DATA_WITH_STRUCT(AllReduceCustomTilingData, tilingData, tilingGM);
    
        Hccl hccl;
        GM_ADDR contextGM = AscendC::GetHcclContext<0>();  // AscendC自定义算子kernel中，通过此方式获取HCCL context
      
        if (AscendC::g_coreType == AIV) {  // 指定AIV核通信   
            hccl.InitV2(contextGM, &tilingData);
            auto ret = hccl.SetCcTilingV2(offsetof(AllReduceCustomTilingData, mc2CcTiling));
            if (ret) {
                return;
            }
            HcclHandle handleId1 = hccl.AllReduce<true>(sendBuf, recvBuf, sendCount, HcclDataType::HCCL_DATA_TYPE_FP16, reduceOp);
            hccl.Wait(handleId1);    
            AscendC::SyncAll<true>();  // 全AIV核同步，防止0核执行过快，提前调用hccl.Finalize()接口，导致其他核Wait卡死   
            hccl.Finalize();
        }
    }
    ```

-   多轮切分场景

    使能多轮切分，等效处理上述非多轮切分示例的通信。如下图所示，每张卡的300个float16数据，被切分为2个首块数据，1个尾块数据。每个首块的数据量tileLen为128个float16数据，尾块的数据量tailLen为44个float16数据。在算子内部实现时，需要对切分后的数据分3轮进行AllReduce通信任务，将等效上述非多轮切分的通信结果。

    **图 3**  各卡数据切分示意图<a name="fig26821908496"></a>  
    <!-- img2text -->
```text
                    rank0 xGM                 rank1 xGM                 rank2 xGM                 rank3 xGM
count{
      ┌────────────────────┐    }tileLen=128  ┌────────────────────┐    ┌────────────────────┐    ┌────────────────────┐
      │        0-0         │                  │        1-0         │    │        2-0         │    │        3-0         │
      ├ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┤                  ├ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┤    ├ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┤    ├ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┤
      │        0-1         │    }tileLen=128  │        1-1         │    │        2-1         │    │        3-1         │
      ├ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┤                  ├ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┤    ├ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┤    ├ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┤
      │        0-2         │    }tailLen=44   │        1-2         │    │        2-2         │    │        3-2         │
      └────────────────────┘                  └────────────────────┘    └────────────────────┘    └────────────────────┘
```

    具体实现为，第1轮通信，每个rank上0-0\\1-0\\2-0\\3-0数据块进行AllReduce处理。第2轮通信，每个rank上0-1\\1-1\\2-1\\3-1数据块进行AllReduce处理。第3轮通信，每个rank上0-2\\1-2\\2-2\\3-2数据块进行AllReduce处理，图示及代码示例如下。

    **图 4**  4卡AllReduce示意图<a name="fig1662341913515"></a>  
    <!-- img2text -->
```text
rank0 xGM                      rank0 yGM                 rank0 xGM                      rank0 yGM                 rank0 xGM                      rank0 yGM
┌──────────────────┐           ┌──────────────────────┐  ┌──────────────────┐           ┌──────────────────────┐  ┌──────────────────┐           ┌──────────────────────┐
│       0-0        │────┐      │ 0-0 + 1-0 + 2-0 + 3-0│  │       0-0        │           │ 0-0 + 1-0 + 2-0 + 3-0│  │       0-0        │           │ 0-0 + 1-0 + 2-0 + 3-0│
├──────────────────┤    ├─tileLen=128                  ├──────────────────┤           ├──────────────────────┤  ├──────────────────┤           ├──────────────────────┤
│       0-1        │    │      │                      │  │       0-1        │           │ 0-1 + 1-1 + 2-1 + 3-1│  │       0-1        │           │ 0-1 + 1-1 + 2-1 + 3-1│
├──────────────────┤    ├─tailLen=44                   ├──────────────────┤           ├──────────────────────┤  ├──────────────────┤           ├──────────────────────┤
│       0-2        │────┘      │                      │  │       0-2        │           │                      │  │       0-2        │           │ 0-2 + 1-2 + 2-2 + 3-2│
└──────────────────┘           └──────────────────────┘  └──────────────────┘           └──────────────────────┘  └──────────────────┘           └──────────────────────┘

rank1 xGM                      rank1 yGM                 rank1 xGM                      rank1 yGM                 rank1 xGM                      rank1 yGM
┌──────────────────┐           ┌──────────────────────┐  ┌──────────────────┐           ┌──────────────────────┐  ┌──────────────────┐           ┌──────────────────────┐
│       1-0        │           │ 0-0 + 1-0 + 2-0 + 3-0│  │       1-0        │           │ 0-0 + 1-0 + 2-0 + 3-0│  │       1-0        │           │ 0-0 + 1-0 + 2-0 + 3-0│
├──────────────────┤           ├──────────────────────┤  ├──────────────────┤           ├──────────────────────┤  ├──────────────────┤           ├──────────────────────┤
│       1-1        │  ───────→ │                      │  │       1-1        │  ───────→ │ 0-1 + 1-1 + 2-1 + 3-1│  │       1-1        │  ───────→ │ 0-1 + 1-1 + 2-1 + 3-1│
├──────────────────┤   第一轮   ├──────────────────────┤  ├──────────────────┤   第二轮   ├──────────────────────┤  ├──────────────────┤   第三轮   ├──────────────────────┤
│       1-2        │ AllReduce │                      │  │       1-2        │ AllReduce │                      │  │       1-2        │ AllReduce │ 0-2 + 1-2 + 2-2 + 3-2│
└──────────────────┘           └──────────────────────┘  └──────────────────┘           └──────────────────────┘  └──────────────────┘           └──────────────────────┘

rank2 xGM                      rank2 yGM                 rank2 xGM                      rank2 yGM                 rank2 xGM                      rank2 yGM
┌──────────────────┐           ┌──────────────────────┐  ┌──────────────────┐           ┌──────────────────────┐  ┌──────────────────┐           ┌──────────────────────┐
│       2-0        │           │ 0-0 + 1-0 + 2-0 + 3-0│  │       2-0        │           │ 0-0 + 1-0 + 2-0 + 3-0│  │       2-0        │           │ 0-0 + 1-0 + 2-0 + 3-0│
├──────────────────┤           ├──────────────────────┤  ├──────────────────┤           ├──────────────────────┤  ├──────────────────┤           ├──────────────────────┤
│       2-1        │           │                      │  │       2-1        │           │ 0-1 + 1-1 + 2-1 + 3-1│  │       2-1        │           │ 0-1 + 1-1 + 2-1 + 3-1│
├──────────────────┤           ├──────────────────────┤  ├──────────────────┤           ├──────────────────────┤  ├──────────────────┤           ├──────────────────────┤
│       2-2        │           │                      │  │       2-2        │           │                      │  │       2-2        │           │ 0-2 + 1-2 + 2-2 + 3-2│
└──────────────────┘           └──────────────────────┘  └──────────────────┘           └──────────────────────┘  └──────────────────┘           └──────────────────────┘

rank3 xGM                      rank3 yGM                 rank3 xGM                      rank3 yGM                 rank3 xGM                      rank3 yGM
┌──────────────────┐           ┌──────────────────────┐  ┌──────────────────┐           ┌──────────────────────┐  ┌──────────────────┐           ┌──────────────────────┐
│       3-0        │           │ 0-0 + 1-0 + 2-0 + 3-0│  │       3-0        │           │ 0-0 + 1-0 + 2-0 + 3-0│  │       3-0        │           │ 0-0 + 1-0 + 2-0 + 3-0│
├──────────────────┤           ├──────────────────────┤  ├──────────────────┤           ├──────────────────────┤  ├──────────────────┤           ├──────────────────────┤
│       3-1        │           │                      │  │       3-1        │           │ 0-1 + 1-1 + 2-1 + 3-1│  │       3-1        │           │ 0-1 + 1-1 + 2-1 + 3-1│
├──────────────────┤           ├──────────────────────┤  ├──────────────────┤           ├──────────────────────┤  ├──────────────────┤           ├──────────────────────┤
│       3-2        │           │                      │  │       3-2        │           │                      │  │       3-2        │           │ 0-2 + 1-2 + 2-2 + 3-2│
└──────────────────┘           └──────────────────────┘  └──────────────────┘           └──────────────────────┘  └──────────────────┘           └──────────────────────┘
```

    ```
    extern "C" __global__ __aicore__ void all_reduce_custom(GM_ADDR xGM, GM_ADDR yGM, GM_ADDR workspaceGM, GM_ADDR tilingGM) {
        constexpr uint32_t tileNum = 2U;   // 首块数量
        constexpr uint64_t tileLen = 128U; // 首块数据个数
        constexpr uint32_t tailNum = 1U;   // 尾块数量
        constexpr uint64_t tailLen = 44U;  // 尾块数据个数
        auto sendBuf = xGM;  // xGM为AllReduce的输入GM地址
        auto recvBuf = yGM;  // yGM为AllReduce的输出GM地址
        HcclReduceOp reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
        REGISTER_TILING_DEFAULT(AllReduceCustomTilingData); //AllReduceCustomTilingData为对应算子头文件定义的结构体
        GET_TILING_DATA_WITH_STRUCT(AllReduceCustomTilingData, tilingData, tilingGM);
    
        Hccl hccl;
        GM_ADDR contextGM = AscendC::GetHcclContext<0>();  // AscendC自定义算子kernel中，通过此方式获取HCCL context
        if (AscendC::g_coreType == AIV) {  // 指定AIV核通信   
            hccl.InitV2(contextGM, &tilingData);
            auto ret = hccl.SetCcTilingV2(offsetof(AllReduceCustomTilingData, mc2CcTiling));
            if (ret != HCCL_SUCCESS) {
                return;
            }
            // 2个首块处理
            constexpr uint32_t tileRepeat = tileNum; 
            // 除了sendBuf和recvBuf入参不同，对2个首块处理的其余参数相同。故使用repeat=2，第2个首块AllReduce任务的sendBuf、recvBuf将由API内部自行更新
            HcclHandle handleId1 = hccl.AllReduce<true>(sendBuf, recvBuf, tileLen, HcclDataType::HCCL_DATA_TYPE_FP16, reduceOp, tileRepeat); 
            // 1个尾块处理
            constexpr uint32_t kSizeOfFloat16 = 2U;
            sendBuf += tileLen * tileNum * kSizeOfFloat16;
            recvBuf += tileLen * tileNum * kSizeOfFloat16;
            constexpr uint32_t tailRepeat = tailNum; 
            HcclHandle handleId2 = hccl.AllReduce<true>(sendBuf, recvBuf, tailLen, HcclDataType::HCCL_DATA_TYPE_FP16, reduceOp, tailRepeat);
    
            for (uint8_t i=0; i<tileRepeat; i++) {
                hccl.Wait(handleId1);
            }
            hccl.Wait(handleId2);  
            AscendC::SyncAll<true>();  // 全AIV核同步，防止0核执行过快，提前调用hccl.Finalize()接口，导致其他核Wait卡死   
            hccl.Finalize();
        }
    }
    ```

