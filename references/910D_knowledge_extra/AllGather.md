# AllGather<a name="ZH-CN_TOPIC_0000002523343908"></a>

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

集合通信算子AllGather的任务下发接口，返回该任务的标识handleId给用户。AllGather的功能为：将通信域内所有节点的输入按照rank id重新排序，然后拼接起来，再将结果发送到所有节点的输出。

<!-- img2text -->
```text
                    rank 0        rank 1        rank 2        rank 3                         rank 0        rank 1        rank 2        rank 3
                  ┆            ┆            ┆            ┆                               ┆            ┆            ┆            ┆
                  ┆  ┌──────┐  ┆            ┆            ┆                               ┆ ┌──────┐   ┆ ┌──────┐   ┆ ┌──────┐   ┆ ┌──────┐
                  ┆  │ in0  │  ┆            ┆            ┆                               ┆ │ out  │   ┆ │ out  │   ┆ │ out  │   ┆ │ out  │
                  ┆  └──────┘  ┆  ┌──────┐  ┆            ┆                               ┆ ├──────┤   ┆ ├──────┤   ┆ ├──────┤   ┆ ├──────┤
                  ┆            ┆  │ in1  │  ┆            ┆         ┌──────────┐          ┆ │      │   ┆ │      │   ┆ │      │   ┆ │      │
                  ┆            ┆  └──────┘  ┆  ┌──────┐  ┆   ───→  │AllGather │  ───→    ┆ ├──────┤   ┆ ├──────┤   ┆ ├──────┤   ┆ ├──────┤
                  ┆            ┆            ┆  │ in2  │  ┆         └──────────┘          ┆ │      │   ┆ │      │   ┆ │      │   ┆ │      │
                  ┆            ┆            ┆  └──────┘  ┆                               ┆ ├──────┤   ┆ ├──────┤   ┆ ├──────┤   ┆ ├──────┤
                  ┆            ┆            ┆            ┆  ┌──────┐                      ┆ │      │   ┆ │      │   ┆ │      │   ┆ │      │
                  ┆            ┆            ┆            ┆  │ in3  │                      ┆ └──────┘   ┆ └──────┘   ┆ └──────┘   ┆ └──────┘
                  ┆            ┆            ┆            ┆  └──────┘                      ┆            ┆            ┆            ┆
```

## 函数原型<a name="section620mcpsimp"></a>

```
template <bool commit = false>
__aicore__ inline HcclHandle AllGather(GM_ADDR sendBuf, GM_ADDR recvBuf, uint64_t sendCount, HcclDataType dataType, uint64_t strideCount, uint8_t repeat = 1)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table149053404318"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000002554424815_zh-cn_topic_0235751031_row27598891"><th class="cellrowborder" valign="top" width="17.77%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000002554424815_zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0000002554424815_zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0000002554424815_zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="11.799999999999999%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000002554424815_zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0000002554424815_zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0000002554424815_zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="70.43%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000002554424815_zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0000002554424815_zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0000002554424815_zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000002554424815_row42461942101815"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000002554424815_p163481714145518"><a name="zh-cn_topic_0000002554424815_p163481714145518"></a><a name="zh-cn_topic_0000002554424815_p163481714145518"></a>commit</p>
</td>
<td class="cellrowborder" valign="top" width="11.799999999999999%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000002554424815_p33487148556"><a name="zh-cn_topic_0000002554424815_p33487148556"></a><a name="zh-cn_topic_0000002554424815_p33487148556"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="70.43%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000002554424815_p186182538493"><a name="zh-cn_topic_0000002554424815_p186182538493"></a><a name="zh-cn_topic_0000002554424815_p186182538493"></a>bool类型。参数取值如下：</p>
<a name="zh-cn_topic_0000002554424815_ul77246714401"></a><a name="zh-cn_topic_0000002554424815_ul77246714401"></a><ul id="zh-cn_topic_0000002554424815_ul77246714401"><li>true：在调用Prepare接口时，Commit同步通知服务端可以执行该通信任务。</li><li>false：在调用Prepare接口时，不通知服务端执行该通信任务。</li></ul>
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
<tr id="row338315247526"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p122031228145219"><a name="p122031228145219"></a><a name="p122031228145219"></a>recvBuf</p>
</td>
<td class="cellrowborder" valign="top" width="11.799999999999999%" headers="mcps1.2.4.1.2 "><p id="p7203142815218"><a name="p7203142815218"></a><a name="p7203142815218"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="70.43%" headers="mcps1.2.4.1.3 "><p id="p112048280529"><a name="p112048280529"></a><a name="p112048280529"></a>目的数据buffer地址，集合通信结果输出到此buffer中。</p>
</td>
</tr>
<tr id="row17902814191610"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p2090218149169"><a name="p2090218149169"></a><a name="p2090218149169"></a>sendCount</p>
</td>
<td class="cellrowborder" valign="top" width="11.799999999999999%" headers="mcps1.2.4.1.2 "><p id="p0902714111616"><a name="p0902714111616"></a><a name="p0902714111616"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="70.43%" headers="mcps1.2.4.1.3 "><p id="p5902191421615"><a name="p5902191421615"></a><a name="p5902191421615"></a>参与AllGather操作的sendBuf的数据个数；recvBuf的数据个数等于sendCount * rank size，即sendCount * 卡数。</p>
</td>
</tr>
<tr id="row48213177160"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p158219172167"><a name="p158219172167"></a><a name="p158219172167"></a>dataType</p>
</td>
<td class="cellrowborder" valign="top" width="11.799999999999999%" headers="mcps1.2.4.1.2 "><p id="p118152714175"><a name="p118152714175"></a><a name="p118152714175"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="70.43%" headers="mcps1.2.4.1.3 "><p id="p98211178163"><a name="p98211178163"></a><a name="p98211178163"></a>AllGather操作的数据类型，目前支持HcclDataType包含的全部数据类型，HcclDataType详细可参考<a href="HCCL使用说明.md#table116710585514">表1</a>。</p>
</td>
</tr>
<tr id="row1481814577161"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p3818657171610"><a name="p3818657171610"></a><a name="p3818657171610"></a>strideCount</p>
</td>
<td class="cellrowborder" valign="top" width="11.799999999999999%" headers="mcps1.2.4.1.2 "><p id="p11490527181714"><a name="p11490527181714"></a><a name="p11490527181714"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="70.43%" headers="mcps1.2.4.1.3 "><a name="ul479232014318"></a><a name="ul479232014318"></a><ul id="ul479232014318"><li>strideCount=0，表示多张卡的数据拼接到一张卡的recvBuf时，相邻数据块保持地址连续。卡rank[i]的数据块将被放在recvBuf中，且偏移数据量为i*sendCount。非多轮切分场景下，推荐用户设置该参数为0。</li><li>strideCount&gt;0，表示多张卡的数据拼接到一张卡的recvBuf时，相邻数据块在recvBuf中起始地址的偏移数据量为strideCount。卡rank[i]的数据块将被放在recvBuf中，且偏移数据量为i*strideCount。</li></ul>
<p id="p19358115315411"><a name="p19358115315411"></a><a name="p19358115315411"></a>注意：上述的偏移数据量为数据个数，单位为sizeof(dataType)。</p>
</td>
</tr>
<tr id="row165671519176"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p17656215111710"><a name="p17656215111710"></a><a name="p17656215111710"></a>repeat</p>
</td>
<td class="cellrowborder" valign="top" width="11.799999999999999%" headers="mcps1.2.4.1.2 "><p id="p38942811174"><a name="p38942811174"></a><a name="p38942811174"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="70.43%" headers="mcps1.2.4.1.3 "><p id="p32565356014"><a name="p32565356014"></a><a name="p32565356014"></a>一次下发的AllGather通信任务个数。repeat取值≥1，默认值为1。当repeat&gt;1时，每个AllGather任务的sendBuf和recvBuf地址由服务端自动算出，计算公式如下：</p>
<p id="p91522315916"><a name="p91522315916"></a><a name="p91522315916"></a>sendBuf[i] = sendBuf + sendCount* sizeof(datatype) * i, i∈[0, repeat)</p>
<p id="p19310955081"><a name="p19310955081"></a><a name="p19310955081"></a>recvBuf[i] = recvBuf + sendCount* sizeof(datatype) * i, i∈[0, repeat)</p>
<p id="p777292611013"><a name="p777292611013"></a><a name="p777292611013"></a>注意：当设置repeat&gt;1时，须与strideCount参数配合使用，规划通信数据地址。</p>
</td>
</tr>
</tbody>
</table>

**图 1**  AllGather通信示例<a name="fig1266464712199"></a>  
<!-- img2text -->
```text
sendCount
  {
┌───────────────┐      ┌───────────────┐      ┌───────────────┐      ┌───────────────┐
│     rank0     │      │     rank1     │      │     rank2     │      │     rank3     │
├───────────────┤      ├───────────────┤      ├───────────────┤      ├───────────────┤
│      0-0      │──────→│      1-0      │──────→│      2-0      │──────┐│      3-0      │
├ ─ ─ ─ ─ ─ ─ ─ ┤      ├ ─ ─ ─ ─ ─ ─ ─ ┤      ├ ─ ─ ─ ─ ─ ─ ─ ┤      ├ ─ ─ ─ ─ ─ ─ ─ ┤
│      0-1      │      │      1-1      │      │      2-1      │──┐   │      3-1      │
├ ─ ─ ─ ─ ─ ─ ─ ┤      ├ ─ ─ ─ ─ ─ ─ ─ ┤      ├ ─ ─ ─ ─ ─ ─ ─ ┤  │   ├ ─ ─ ─ ─ ─ ─ ─ ┤
│      0-2      │      │      1-2      │      │      2-2      │  └──→│      3-2      │
└───────────────┘      └───────────────┘      └───────────────┘      └───────────────┘
  }

                                                                    ┌────────────┐
                                                                    │  第一轮    │
                                      ┌────────────────────────────→│    All     │───→┌───────────────┐
                                      │                             │  Gather    │    │      0-0      │
                                      │                             └────────────┘    └───────────────┘
                                      │                                                  } strideCount
                                      ├────────────────────────────────────────────────→┌───────────────┐
                                      │                                                │      1-0      │
                                      │                                                └───────────────┘
                                      ├────────────────────────────────────────────────→┌───────────────┐
                                      │                                                │      2-0      │
                                      │                                                └───────────────┘
                                      └────────────────────────────────────────────────→┌───────────────┐
                                                                                       │      3-0      │
                                                                                       └───────────────┘

                                                                              rank0 recvBuf
                                                                        ( AllGather第1轮结果 )
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

    如下图所示，4张卡上均有sendCount=300个float16数据，每张卡从xGM内存中获取到本卡数据，gather处理各卡的数据后，将结果输出到各卡的yGM。

    **图 2**  非多轮切分场景下4卡AllGather通信<a name="fig13621175034619"></a>  
    <!-- img2text -->
```text
sendBuf
  │
  │ sendCount
  ▼
        rank0           rank1           rank2           rank3
   ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐
   │    0-0     │  │    1-0     │  │    2-0     │  │    3-0     │
   └────────────┘  └────────────┘  └────────────┘  └────────────┘

                    AllGather
                       │
                       ▼

recvBuf
                         rank0           rank1           rank2           rank3
                    ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐
                    │    0-0     │  │    0-0     │  │    0-0     │  │    0-0     │
                    ├────────────┤  ├────────────┤  ├────────────┤  ├────────────┤
                    │    1-0     │  │    1-0     │  │    1-0     │  │    1-0     │
                    ├────────────┤  ├────────────┤  ├────────────┤  ├────────────┤
                    │    2-0     │  │    2-0     │  │    2-0     │  │    2-0     │
                    ├────────────┤  ├────────────┤  ├────────────┤  ├────────────┤
                    │    3-0     │  │    3-0     │  │    3-0     │  │    3-0     │
                    └────────────┘  └────────────┘  └────────────┘  └────────────┘
```

    ```
    extern "C" __global__ __aicore__ void all_gather_custom(GM_ADDR xGM, GM_ADDR yGM, GM_ADDR workspaceGM, GM_ADDR tilingGM) {
        auto sendBuf = xGM;  // xGM为AllGather的输入GM地址
        auto recvBuf = yGM;  // yGM为AllGather的输出GM地址
        uint64_t sendCount = 300;  // 每张卡均有300个float16的数据
        uint64_t strideCount = 0;  // 非切分场景strideCount可设置为0
        REGISTER_TILING_DEFAULT(AllGatherCustomTilingData); //AllGatherCustomTilingData为对应算子头文件定义的结构体
        GET_TILING_DATA_WITH_STRUCT(AllGatherCustomTilingData, tilingData, tilingGM);
    
        Hccl hccl;
        GM_ADDR contextGM = AscendC::GetHcclContext<0>();  // AscendC自定义算子kernel中，通过此方式获取HCCL context
    
        if (AscendC::g_coreType == AIV) {  // 指定AIV核通信   
            hccl.InitV2(contextGM, &tilingData);
            auto ret = hccl.SetCcTilingV2(offsetof(AllGatherCustomTilingData, allGatherCcTiling));
            if (ret != HCCL_SUCCESS) {
              return;
            }
            HcclHandle handleId1 = hccl.AllGather<true>(sendBuf, recvBuf, sendCount, HcclDataType::HCCL_DATA_TYPE_FP16, strideCount);
            hccl.Wait(handleId1);    
            AscendC::SyncAll<true>();  // 全AIV核同步，防止0核执行过快，提前调用hccl.Finalize()接口，导致其他核Wait卡死   
            hccl.Finalize();
        }
    }
    ```

-   多轮切分场景

    使能多轮切分，等效处理上述非多轮切分示例的通信。如下图所示，每张卡的300个float16数据，被切分为2个首块数据，1个尾块数据。每个首块的数据量tileLen为128个float16数据，尾块的数据量tailLen为44个float16数据。在算子内部实现时，需要对切分后的数据分3轮进行AllGather通信任务，将等效上述非多轮切分的通信结果。

    **图 3**  各卡数据切分示意图<a name="fig26821908496"></a>  
    <!-- img2text -->
```
                    rank0               rank1               rank2               rank3

tileLen=128  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
tileLen=128  │     0-0      │    │     1-0      │    │     2-0      │    │     3-0      │
             ├──────────────┤    ├──────────────┤    ├──────────────┤    ├──────────────┤
             │     0-1      │    │     1-1      │    │     2-1      │    │     3-1      │
tailLen=44   ├──────────────┤    ├──────────────┤    ├──────────────┤    ├──────────────┤
             │     0-2      │    │     1-2      │    │     2-2      │    │     3-2      │
             └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

    具体实现为，第1轮通信，每个rank上0-0\\1-0\\2-0\\3-0数据块进行AllGather处理。第2轮通信，每个rank上0-1\\1-1\\2-1\\3-1数据块进行AllGather处理。第3轮通信，每个rank上0-2\\1-2\\2-2\\3-2数据块进行AllGather处理。每一轮通信结果中，各卡上相邻数据块的起始地址间隔的数据个数为strideCount，以第一轮通信结果为例，rank0的0-0数据块和1-0数据块起始地址间隔的数据量strideCount = 2\*tileLen+1\*tailLen=300。

    **图 4**  第一轮4卡AllGather示意图<a name="fig1662341913515"></a>  
    <!-- img2text -->
```text
xGM
tileLen  {                    rank0                 rank1                 rank2                 rank3
tailLen  {              ┌────────────┐       ┌────────────┐       ┌────────────┐       ┌────────────┐
                        │    0-0     │       │    1-0     │       │    2-0     │       │    3-0     │
                        ├────────────┤       ├────────────┤       ├────────────┤       ├────────────┤
                        │    0-1     │       │    1-1     │       │    2-1     │       │    3-1     │
                        ├────────────┤       ├────────────┤       ├────────────┤       ├────────────┤
                        │    0-2     │       │    1-2     │       │    2-2     │       │    3-2     │
                        └────────────┘       └────────────┘       └────────────┘       └────────────┘

                                                  AllGather
                                             ─────────────────→

yGM                 rank0                 rank1                 rank2                 rank3
              ┌────────────┐       ┌────────────┐       ┌────────────┐       ┌────────────┐
              │    0-0     │       │    0-0     │       │    0-0     │       │    0-0     │
              ├────────────┤       ├────────────┤       ├────────────┤       ├────────────┤
              │    0-1     │       │    0-1     │       │    0-1     │       │    0-1     │
              ├────────────┤       ├────────────┤       ├────────────┤       ├────────────┤
              │    0-2     │       │    0-2     │       │    0-2     │       │    0-2     │
              ├────────────┤       ├────────────┤       ├────────────┤       ├────────────┤
              │    1-0     │       │    1-0     │       │    1-0     │       │    1-0     │
              ├────────────┤       ├────────────┤       ├────────────┤       ├────────────┤
              │    1-1     │       │    1-1     │       │    1-1     │       │    1-1     │
              ├────────────┤       ├────────────┤       ├────────────┤       ├────────────┤
              │    1-2     │       │    1-2     │       │    1-2     │       │    1-2     │
              ├────────────┤       ├────────────┤       ├────────────┤       ├────────────┤
              │    2-0     │       │    2-0     │       │    2-0     │       │    2-0     │
              ├────────────┤       ├────────────┤       ├────────────┤       ├────────────┤
              │    2-1     │       │    2-1     │       │    2-1     │       │    2-1     │
              ├────────────┤       ├────────────┤       ├────────────┤       ├────────────┤
              │    2-2     │       │    2-2     │       │    2-2     │       │    2-2     │
              ├────────────┤       ├────────────┤       ├────────────┤       ├────────────┤
              │    3-0     │       │    3-0     │       │    3-0     │       │    3-0     │
              ├────────────┤       ├────────────┤       ├────────────┤       ├────────────┤
              │    3-1     │       │    3-1     │       │    3-1     │       │    3-1     │
              ├────────────┤       ├────────────┤       ├────────────┤       ├────────────┤
              │    3-2     │       │    3-2     │       │    3-2     │       │    3-2     │
              └────────────┘       └────────────┘       └────────────┘       └────────────┘
                   ↑
                   │
                   │ strideCount
                   ↓
```

说明:
- 左侧 xGM 中，每个 rank 含 3 个数据块：首块为 `*-0`，其长度对应 `tileLen`；后两个尾块为 `*-1`、`*-2`，其长度对应 `tailLen`
- 图题对应：第一轮4卡AllGather示意图
- AllGather 后，`yGM` 中每个 rank 的结果相同，按 `0-0/0-1/0-2/1-0/1-1/1-2/2-0/2-1/2-2/3-0/3-1/3-2` 顺序排列
- `strideCount` 标注的是 AllGather 结果中相邻首块起始地址之间间隔的数据个数；图中示意的是 `0-0` 与 `1-0` 之间的间隔范围

    ```
    extern "C" __global__ __aicore__ void all_gather_custom(GM_ADDR xGM, GM_ADDR yGM, GM_ADDR workspaceGM, GM_ADDR tilingGM) {
        constexpr uint32_t tileNum = 2U;   // 首块数量
        constexpr uint64_t tileLen = 128U; // 首块数据个数
        constexpr uint32_t tailNum = 1U;   // 尾块数量
        constexpr uint64_t tailLen = 44U;  // 尾块数据个数
        auto sendBuf = xGM;  // xGM为AllGather的输入GM地址
        auto recvBuf = yGM;  // yGM为AllGather的输出GM地址
        REGISTER_TILING_DEFAULT(AllGatherCustomTilingData); //AllGatherCustomTilingData为对应算子头文件定义的结构体
        GET_TILING_DATA_WITH_STRUCT(AllGatherCustomTilingData, tilingData, tilingGM);
    
        Hccl hccl;
        GM_ADDR contextGM = AscendC::GetHcclContext<0>();  // AscendC自定义算子kernel中，通过此方式获取HCCL context
        if (AscendC::g_coreType == AIV) {  // 指定AIV核通信   
            hccl.InitV2(contextGM, &tilingData);
            auto ret = hccl.SetCcTilingV2(offsetof(AllGatherCustomTilingData, allGatherCcTiling));
            if (ret != HCCL_SUCCESS) {
              return;
            }
            uint64_t strideCount = tileLen * tileNum + tailLen * tailNum;
            // 2个首块处理
            constexpr uint32_t tileRepeat = tileNum; 
            // 除了sendBuf和recvBuf入参不同，处理2个首块的其余参数相同。故使用repeat=2，第2个首块AllGather任务的sendBuf、recvBuf将由API内部自行更新
            HcclHandle handleId1 = hccl.AllGather<true>(sendBuf, recvBuf, tileLen, HcclDataType::HCCL_DATA_TYPE_FP16, strideCount, tileRepeat); 
            // 1个尾块处理
            constexpr uint32_t kSizeOfFloat16 = 2U;
            sendBuf += tileLen * tileNum * kSizeOfFloat16;
            recvBuf += tileLen * tileNum * kSizeOfFloat16;
            constexpr uint32_t tailRepeat = tailNum; 
            HcclHandle handleId2 = hccl.AllGather<true>(sendBuf, recvBuf, tailLen, HcclDataType::HCCL_DATA_TYPE_FP16, strideCount, tailRepeat);
               
            for (uint8_t i=0; i<tileRepeat; i++) {
                hccl.Wait(handleId1);
            }
            hccl.Wait(handleId2);  
            AscendC::SyncAll<true>();  // 全AIV核同步，防止0核执行过快，提前调用hccl.Finalize()接口，导致其他核Wait卡死   
            hccl.Finalize();
        }
    }
    ```

