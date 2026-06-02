# 设置Kernel类型<a name="ZH-CN_TOPIC_0000002554344891"></a>

## 产品支持情况<a name="section1550532418810"></a>

<a name="table38301303189"></a>
<table><thead align="left"><tr id="row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p1883113061818"><a name="p1883113061818"></a><a name="p1883113061818"></a><span id="ph20833205312295"><a name="ph20833205312295"></a><a name="ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p783113012187"><a name="p783113012187"></a><a name="p783113012187"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row1272474920205"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p17301775812"><a name="p17301775812"></a><a name="p17301775812"></a><span id="ph2272194216543"><a name="ph2272194216543"></a><a name="ph2272194216543"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p37256491200"><a name="p37256491200"></a><a name="p37256491200"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="zh-cn_topic_0000001610027821_section212607105720"></a>

用于用户自定义设置kernel类型，控制算子执行时只启动该类型的核，避免启动不需要工作的核，缩短核启动开销。

## 函数原型<a name="zh-cn_topic_0000001610027821_section1630753514297"></a>

-   设置全局默认的kernel type，对所有的tiling key生效。

    当前支持在自定义算子工程和Kernel直调工程中使用。

    ```
    KERNEL_TASK_TYPE_DEFAULT(value)
    ```

-   设置某一个具体的tiling key对应的kernel type。

    当前仅支持在自定义算子工程中使用。

    ```
    KERNEL_TASK_TYPE(key, value)
    ```

## 参数说明<a name="zh-cn_topic_0000001610027821_section129451113125413"></a>

**表 1**  参数说明

<a name="table89201718635"></a>
<table><thead align="left"><tr id="row1992019181938"><th class="cellrowborder" valign="top" width="17.22%" id="mcps1.2.4.1.1"><p id="p109201118735"><a name="p109201118735"></a><a name="p109201118735"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="15.340000000000002%" id="mcps1.2.4.1.2"><p id="p8920018334"><a name="p8920018334"></a><a name="p8920018334"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="67.44%" id="mcps1.2.4.1.3"><p id="p199211318136"><a name="p199211318136"></a><a name="p199211318136"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row692113181531"><td class="cellrowborder" valign="top" width="17.22%" headers="mcps1.2.4.1.1 "><p id="p1392117186310"><a name="p1392117186310"></a><a name="p1392117186310"></a>key</p>
</td>
<td class="cellrowborder" valign="top" width="15.340000000000002%" headers="mcps1.2.4.1.2 "><p id="p16921418832"><a name="p16921418832"></a><a name="p16921418832"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.44%" headers="mcps1.2.4.1.3 "><p id="p12921131811315"><a name="p12921131811315"></a><a name="p12921131811315"></a>tiling key的key值，此参数是正数，表示某个核函数的分支。</p>
</td>
</tr>
<tr id="row1792151817316"><td class="cellrowborder" valign="top" width="17.22%" headers="mcps1.2.4.1.1 "><p id="p119219181935"><a name="p119219181935"></a><a name="p119219181935"></a>value</p>
</td>
<td class="cellrowborder" valign="top" width="15.340000000000002%" headers="mcps1.2.4.1.2 "><p id="p18921181812316"><a name="p18921181812316"></a><a name="p18921181812316"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.44%" headers="mcps1.2.4.1.3 "><p id="p14921918033"><a name="p14921918033"></a><a name="p14921918033"></a>设置的kernel类型，可选值范围，kernel类型具体说明请参考<a href="#table76335324910">表2</a>。不同硬件架构支持的参数取值不同，具体支持的参数取值请参考<a href="#li693212153417">kernel&nbsp;type取值约束</a>。</p>
<a name="screen2031620461932"></a><a name="screen2031620461932"></a><pre class="screen" codetype="Cpp" id="screen2031620461932">enum KernelMetaType {
    KERNEL_TYPE_AIV_ONLY,
    KERNEL_TYPE_AIC_ONLY,
    KERNEL_TYPE_MIX_AIV_1_0,
    KERNEL_TYPE_MIX_AIC_1_0,
    KERNEL_TYPE_MIX_AIC_1_1,
    KERNEL_TYPE_MIX_AIC_1_2,
<span>    KERNEL_TYPE_AICORE</span>,
<span>    KERNEL_TYPE_VECTORCORE,</span>
<span>    KERNEL_TYPE_MIX_AICORE,</span>
<span>    KERNEL_TYPE_MIX_VECTOR_CORE</span>,
<span>    KERNEL_TYPE_MAX</span>
};</pre>
</td>
</tr>
</tbody>
</table>

**表 2**  kernel type取值说明

<a name="table76335324910"></a>
<table><thead align="left"><tr id="row13633133211918"><th class="cellrowborder" valign="top" width="35.8%" id="mcps1.2.3.1.1"><p id="p196346329913"><a name="p196346329913"></a><a name="p196346329913"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="64.2%" id="mcps1.2.3.1.2"><p id="p1163403212911"><a name="p1163403212911"></a><a name="p1163403212911"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row1663413326917"><td class="cellrowborder" valign="top" width="35.8%" headers="mcps1.2.3.1.1 "><p id="p12634173214918"><a name="p12634173214918"></a><a name="p12634173214918"></a>KERNEL_TYPE_AIV_ONLY</p>
</td>
<td class="cellrowborder" valign="top" width="64.2%" headers="mcps1.2.3.1.2 "><p id="p14634183212912"><a name="p14634183212912"></a><a name="p14634183212912"></a>算子执行时仅启动AI Core上的Vector核：比如用户在host侧设置numBlocks为10，则会启动10个Vector核。</p>
</td>
</tr>
<tr id="row166346321898"><td class="cellrowborder" valign="top" width="35.8%" headers="mcps1.2.3.1.1 "><p id="p1863413321917"><a name="p1863413321917"></a><a name="p1863413321917"></a>KERNEL_TYPE_AIC_ONLY</p>
</td>
<td class="cellrowborder" valign="top" width="64.2%" headers="mcps1.2.3.1.2 "><p id="p17634113218918"><a name="p17634113218918"></a><a name="p17634113218918"></a>算子执行时仅启动AI Core上的Cube核：比如用户在host侧设置numBlocks为10，则会启动10个Cube核。</p>
</td>
</tr>
<tr id="row196349321693"><td class="cellrowborder" valign="top" width="35.8%" headers="mcps1.2.3.1.1 "><p id="p1863411329915"><a name="p1863411329915"></a><a name="p1863411329915"></a>KERNEL_TYPE_MIX_AIV_1_0</p>
</td>
<td class="cellrowborder" valign="top" width="64.2%" headers="mcps1.2.3.1.2 "><p id="p101533312255"><a name="p101533312255"></a><a name="p101533312255"></a>AIC、AIV混合场景下，使用了<a href="核间同步.md">多核控制相关指令</a>时，设置核函数的类型为MIX AIV:AIC 1:0（带有硬同步），算子执行时仅会启动AI Core上的Vector核，比如用户在host侧设置numBlocks为10，则会启动10个Vector核。</p>
<p id="p15178718334"><a name="p15178718334"></a><a name="p15178718334"></a>硬同步的概念解释如下：当不同核之间操作同一块全局内存且可能存在读后写、写后读以及写后写等数据依赖问题时，通过调用<a href="SyncAll.md">SyncAll()</a>函数来插入同步语句来避免上述数据依赖时可能出现的数据读写错误问题。目前多核同步分为硬同步和软同步，硬同步是利用硬件自带的全核同步指令由硬件保证多核同步。</p>
</td>
</tr>
<tr id="row12635183213919"><td class="cellrowborder" valign="top" width="35.8%" headers="mcps1.2.3.1.1 "><p id="p7635932295"><a name="p7635932295"></a><a name="p7635932295"></a>KERNEL_TYPE_MIX_AIC_1_0</p>
</td>
<td class="cellrowborder" valign="top" width="64.2%" headers="mcps1.2.3.1.2 "><p id="p121619511252"><a name="p121619511252"></a><a name="p121619511252"></a>AIC、AIV混合场景下，使用了<a href="核间同步.md">多核控制相关指令</a>时，设置核函数的类型为MIX AIC:AIV 1:0（带有硬同步），算子执行时仅会启动AI Core上的Cube核，比如用户在host侧设置numBlocks为10，则会启动10个Cube核。</p>
</td>
</tr>
<tr id="row76354321096"><td class="cellrowborder" valign="top" width="35.8%" headers="mcps1.2.3.1.1 "><p id="p146354324910"><a name="p146354324910"></a><a name="p146354324910"></a>KERNEL_TYPE_MIX_AIC_1_1</p>
</td>
<td class="cellrowborder" valign="top" width="64.2%" headers="mcps1.2.3.1.2 "><p id="p1763563216919"><a name="p1763563216919"></a><a name="p1763563216919"></a>AIC、AIV混合场景下，设置核函数的类型为MIX AIC:AIV 1:1，算子执行时会同时启动AI Core上的Cube核和Vector核，比如用户在host侧设置numBlocks为10，则会启动10个Cube核和10个Vector核。</p>
</td>
</tr>
<tr id="row15635332191"><td class="cellrowborder" valign="top" width="35.8%" headers="mcps1.2.3.1.1 "><p id="p13635113220918"><a name="p13635113220918"></a><a name="p13635113220918"></a>KERNEL_TYPE_MIX_AIC_1_2</p>
</td>
<td class="cellrowborder" valign="top" width="64.2%" headers="mcps1.2.3.1.2 "><p id="p16635332397"><a name="p16635332397"></a><a name="p16635332397"></a>AIC、AIV混合场景下，设置核函数的类型为MIX AIC:AIV 1:2，算子执行时会同时启动AI Core上的Cube核和Vector核，比如用户在host侧设置numBlocks为10，则会启动10个Cube核和20个Vector核。</p>
</td>
</tr>
<tr id="row66355321498"><td class="cellrowborder" valign="top" width="35.8%" headers="mcps1.2.3.1.1 "><p id="p863516323915"><a name="p863516323915"></a><a name="p863516323915"></a><span>KERNEL_TYPE_AICORE </span></p>
</td>
<td class="cellrowborder" valign="top" width="64.2%" headers="mcps1.2.3.1.2 "><p id="p1990333814243"><a name="p1990333814243"></a><a name="p1990333814243"></a>算子执行时仅会启动AI Core，比如用户在host侧设置numBlocks为5，则会启动5个AI Core。</p>
</td>
</tr>
<tr id="row86359321091"><td class="cellrowborder" valign="top" width="35.8%" headers="mcps1.2.3.1.1 "><p id="p363511321396"><a name="p363511321396"></a><a name="p363511321396"></a><span>KERNEL_TYPE_VECTORCORE</span></p>
</td>
<td class="cellrowborder" valign="top" width="64.2%" headers="mcps1.2.3.1.2 "><p id="p82581757123"><a name="p82581757123"></a><a name="p82581757123"></a>该参数为预留参数，当前版本暂不支持。</p>
</td>
</tr>
<tr id="row2635113215912"><td class="cellrowborder" valign="top" width="35.8%" headers="mcps1.2.3.1.1 "><p id="p18921145171312"><a name="p18921145171312"></a><a name="p18921145171312"></a><span>KERNEL_TYPE_MIX_AICORE</span></p>
</td>
<td class="cellrowborder" valign="top" width="64.2%" headers="mcps1.2.3.1.2 "><p id="p1842315851216"><a name="p1842315851216"></a><a name="p1842315851216"></a>该参数为预留参数，当前版本暂不支持。</p>
</td>
</tr>
<tr id="row3636193212912"><td class="cellrowborder" valign="top" width="35.8%" headers="mcps1.2.3.1.1 "><p id="p663633211910"><a name="p663633211910"></a><a name="p663633211910"></a><span>KERNEL_TYPE_MIX_VECTOR_CORE</span></p>
</td>
<td class="cellrowborder" valign="top" width="64.2%" headers="mcps1.2.3.1.2 "><p id="p1487434912401"><a name="p1487434912401"></a><a name="p1487434912401"></a>基于Ascend C开发的矢量计算相关的算子可以运行在<span id="ph182831320115814"><a name="ph182831320115814"></a><a name="ph182831320115814"></a>Vector Core</span>上，调用本接口传入该参数用于使能<span id="ph12899342131115"><a name="ph12899342131115"></a><a name="ph12899342131115"></a>Vector Core</span>。</p>
<p id="p14506850174012"><a name="p14506850174012"></a><a name="p14506850174012"></a><span id="ph5667172611155"><a name="ph5667172611155"></a><a name="ph5667172611155"></a>使能<span id="ph19742192610399"><a name="ph19742192610399"></a><a name="ph19742192610399"></a>Vector Core</span>后，算子执行时会同时启动AI Core和Vector Core，用于并行计算。比如用户在host侧设置numBlocks为10，则会启动总数为10的AI Core和Vector Core。</span></p>
<p id="p1166792611510"><a name="p1166792611510"></a><a name="p1166792611510"></a>需要注意的是，通过SetBlockDim设置核数时，需要大于AI Core的核数，否则不会启动VectorCore。</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="zh-cn_topic_0000001610027821_section65498832"></a>

-   <a name="li693212153417"></a>kernel type取值约束
    -   Ascend 950PR/Ascend 950DT，支持KERNEL\_TYPE\_AIV\_ONLY、 KERNEL\_TYPE\_AIC\_ONLY、KERNEL\_TYPE\_MIX\_AIV\_1\_0、KERNEL\_TYPE\_MIX\_AIC\_1\_0、KERNEL\_TYPE\_MIX\_AIC\_1\_1、KERNEL\_TYPE\_MIX\_AIC\_1\_2。

-   **KERNEL\_TASK\_TYPE**优先级高于**KERNEL\_TASK\_TYPE\_DEFAULT**，同时设置了全局kernel type和某一个tiling key的kernel type，该tiling key的kernel type以**KERNEL\_TASK\_TYPE**设置的为准。
-   没有设置全局默认kernel type的情况下，如果开发者只为其中的某几个tiling key设置kernel type，即部分tiling key没有设置kernel type，会导致算子kernel编译报错。
-   当设置具体的kernel task type时，用户的算子实现需要与kernel type相匹配。比如用户设置kernel type为KERNEL\_TYPE\_MIX\_AIC\_1\_2，则算子内部实现应与核配比AIC:AIV为1:2相对应；若用户设置kernel type为KERNEL\_TYPE\_AIC\_ONLY， 则算子内部实现应该为纯cube逻辑，不应该存在vector部分的逻辑。其他的kernel type类似。
-   当纯cube或者纯vec算子强制设定kernel type为MIX类型时，workspace的大小不能设置为0，需要设置一个大于0的值（比如16、32等）。
-   使用[Tiling模板编程](Tiling模板编程.md)时，需要通过ASCENDC\_TPL\_KERNEL\_TYPE\_SEL设置Kernel类型即可，无需再通过该接口进行设置，本接口不生效。

## 调用示例<a name="zh-cn_topic_0000001610027821_section97001499599"></a>

-   示例一：使能VectorCore样例
    1.  完成算子kernel侧开发时，需要通过本接口使能Vector Core，算子执行时会同时启动AI Core和Vector Core， 此时AI Core会当成Vector Core使用。示例如下：

        ```
        extern "C" __global__ __aicore__ void add_custom(__gm__ uint8_t *x, __gm__ uint8_t *y, __gm__ uint8_t *z, __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
        {
            GET_TILING_DATA(tilingData, tiling);
            if (workspace == nullptr) {
                return;
            }
            KernelAdd op;
            op.Init(x, y, z, tilingData.numBlocks, tilingData.totalLength, tilingData.tileNum);
            KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_VECTOR_CORE); // 使能VectorCore
            if (TILING_KEY_IS(1)) {
                op.Process1();
            } else if (TILING_KEY_IS(2)) {
                op.Process2();
            }
            // ...
        }
        ```

    2.  完成算子host侧Tiling开发时，设置的numBlocks代表的是AI Core和Vector Core的总数，比如用户在host侧设置numBlocks为10，则会启动总数为10的AI Core和Vector Core；为保证启动Vector Core，设置数值应大于AI Core的核数。您可以通过[GetCoreNumAic](GetCoreNumAic.md)接口获取AI Core的核数，[GetCoreNumVector](GetCoreNumVector.md)接口获取Vector Core的核数。 如下代码片段，展示了numBlocks的设置方法，此处设置为AI Core和Vector Core的总和，表示所有AI Core和Vector Core都启动。

        ```
        // 配套的host侧tiling函数示例：
        ge::graphStatus TilingFunc(gert::TilingContext* context)
        {	
            // 使能VectorCore，将numBlocks置为AI Core中vector核数 + Vector Core中的vector核数
            auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
            auto totalCoreNum = ascendcPlatform.GetCoreNumAiv();
            // ASCENDXXX请替换为实际的版本型号
            if (ascendcPlatform.GetSocVersion() == platform_ascendc::SocVersion::ASCENDXXX) {
               totalCoreNum = totalCoreNum + ascendcPlatform.GetCoreNumVector();
            }
            context->SetBlockDim(totalCoreNum);
        }
        ```

-   示例二：设置某一个具体的tiling key对应的kernel type。如下代码为伪代码 ，不可直接运行。

    ```
    extern "C" __global__ __aicore__ void add_custom(__gm__ uint8_t *x, __gm__ uint8_t *y, __gm__ uint8_t *z, __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
    {
        GET_TILING_DATA(tilingData, tiling);
        if (workspace == nullptr) {
            return;
        }
        KernelAdd op;
        op.Init(x, y, z, tilingData.numBlocks, tilingData.totalLength, tilingData.tileNum);
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY); // 设置默认的kernel类型为纯AIV类型
        if (TILING_KEY_IS(1)) {
            KERNEL_TASK_TYPE(1, KERNEL_TYPE_MIX_AIV_1_0); // 设置tiling key=1对应的kernel类型为MIX AIV 1:0
            op.Process1();
        } else if (TILING_KEY_IS(2)) {
            KERNEL_TASK_TYPE(2, KERNEL_TYPE_AIV_ONLY); // 设置tiling key=2对应的kernel类型为纯AIV类型
            op.Process2();
        }
        // ...
    }
    // 配套的host侧tiling函数示例：
    ge::graphStatus TilingFunc(gert::TilingContext* context)
    {	
        // ...
        if (context->GetInputShape(0) > 10) {
            context->SetTilingKey(1);
        } else if (some condition) {
            context->SetTilingKey(2);
        }
    }
    ```

