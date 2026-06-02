# Copy<a name="ZH-CN_TOPIC_0000002554344793"></a>

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

## 功能说明<a name="section618mcpsimp"></a>

VECIN，VECCALC，VECOUT之间的搬运指令，支持mask操作和DataBlock间隔操作。

## 函数原型<a name="section620mcpsimp"></a>

-   tensor前n个数据计算

    ```
    template <typename T, bool isSetMask = true>
    __aicore__ inline void Copy(const LocalTensor<T>& dst, const LocalTensor<T>& src, const uint32_t count)
    ```

-   tensor高维切分计算
    -   mask逐bit模式

        ```
        template <typename T, bool isSetMask = true>
        __aicore__ inline void Copy(const LocalTensor<T>& dst, const LocalTensor<T>& src, const uint64_t mask[], const uint8_t repeatTime, const CopyRepeatParams& repeatParams)
        ```

    -   mask连续模式

        ```
        template <typename T, bool isSetMask = true>
        __aicore__ inline void Copy(const LocalTensor<T>& dst, const LocalTensor<T>& src, const uint64_t mask, const uint8_t repeatTime, const CopyRepeatParams& repeatParams)
        ```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table4835205712588"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001429830437_row118356578583"><th class="cellrowborder" valign="top" width="15.83%" id="mcps1.2.3.1.1"><p id="zh-cn_topic_0000001429830437_p48354572582"><a name="zh-cn_topic_0000001429830437_p48354572582"></a><a name="zh-cn_topic_0000001429830437_p48354572582"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="84.17%" id="mcps1.2.3.1.2"><p id="zh-cn_topic_0000001429830437_p583535795817"><a name="zh-cn_topic_0000001429830437_p583535795817"></a><a name="zh-cn_topic_0000001429830437_p583535795817"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001429830437_row1835857145817"><td class="cellrowborder" valign="top" width="15.83%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001429830437_p5835457165816"><a name="zh-cn_topic_0000001429830437_p5835457165816"></a><a name="zh-cn_topic_0000001429830437_p5835457165816"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="84.17%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001429830437_p168351657155818"><a name="zh-cn_topic_0000001429830437_p168351657155818"></a><a name="zh-cn_topic_0000001429830437_p168351657155818"></a>操作数数据类型。</p>
<p id="p8993356112520"><a name="p8993356112520"></a><a name="p8993356112520"></a><span id="ph126252025205"><a name="ph126252025205"></a><a name="ph126252025205"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：uint8_t/int8_t/hifloat8_t/fp8_e4m3fn_t/fp8_e5m2_t/fp4x2_e2m1_t/fp4x2_e1m2_t/fp8_e8m0_t/uint16_t/int16_t/half/bfloat16_t/float/uint32_t/int32_t/uint64_t/int64_t</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001429830437_row18835145716587"><td class="cellrowborder" valign="top" width="15.83%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001429830437_p1383515717581"><a name="zh-cn_topic_0000001429830437_p1383515717581"></a><a name="zh-cn_topic_0000001429830437_p1383515717581"></a>isSetMask</p>
</td>
<td class="cellrowborder" valign="top" width="84.17%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002554343931_p77520541653"><a name="zh-cn_topic_0000002554343931_p77520541653"></a><a name="zh-cn_topic_0000002554343931_p77520541653"></a>是否在接口内部设置mask。</p>
<a name="zh-cn_topic_0000002554343931_ul1163765616511"></a><a name="zh-cn_topic_0000002554343931_ul1163765616511"></a><ul id="zh-cn_topic_0000002554343931_ul1163765616511"><li>true，表示在接口内部设置mask。</li><li>false，表示在接口外部设置mask，开发者需要使用<a href="SetVectorMask.md">SetVectorMask</a>接口设置mask值。这种模式下，本接口入参中的mask值必须设置为占位符MASK_PLACEHOLDER。</li></ul>
</td>
</tr>
</tbody>
</table>

**表 2**  参数说明

<a name="table1055216132132"></a>
<table><thead align="left"><tr id="row105531513121315"><th class="cellrowborder" valign="top" width="16.49%" id="mcps1.2.4.1.1"><p id="p5553171319138"><a name="p5553171319138"></a><a name="p5553171319138"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="11.93%" id="mcps1.2.4.1.2"><p id="p5553151313131"><a name="p5553151313131"></a><a name="p5553151313131"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="71.58%" id="mcps1.2.4.1.3"><p id="p655316136139"><a name="p655316136139"></a><a name="p655316136139"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row5553201314135"><td class="cellrowborder" valign="top" width="16.49%" headers="mcps1.2.4.1.1 "><p id="p8553813111314"><a name="p8553813111314"></a><a name="p8553813111314"></a>dst</p>
</td>
<td class="cellrowborder" valign="top" width="11.93%" headers="mcps1.2.4.1.2 "><p id="p755318134134"><a name="p755318134134"></a><a name="p755318134134"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="71.58%" headers="mcps1.2.4.1.3 "><p id="p65530137137"><a name="p65530137137"></a><a name="p65530137137"></a>目的操作数。</p>
<p id="p16703131355116"><a name="p16703131355116"></a><a name="p16703131355116"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。起始地址需要保证32字节对齐。</p>
</td>
</tr>
<tr id="row6553613191315"><td class="cellrowborder" valign="top" width="16.49%" headers="mcps1.2.4.1.1 "><p id="p195531113161311"><a name="p195531113161311"></a><a name="p195531113161311"></a>src</p>
</td>
<td class="cellrowborder" valign="top" width="11.93%" headers="mcps1.2.4.1.2 "><p id="p155310135134"><a name="p155310135134"></a><a name="p155310135134"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.58%" headers="mcps1.2.4.1.3 "><p id="p158685516535"><a name="p158685516535"></a><a name="p158685516535"></a>源操作数。</p>
<p id="p138399710538"><a name="p138399710538"></a><a name="p138399710538"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。起始地址需要保证32字节对齐。</p>
<p id="p1955311137135"><a name="p1955311137135"></a><a name="p1955311137135"></a>源操作数的数据类型需要与目的操作数保持一致。</p>
</td>
</tr>
<tr id="row591792253816"><td class="cellrowborder" valign="top" width="16.49%" headers="mcps1.2.4.1.1 "><p id="p1917202214387"><a name="p1917202214387"></a><a name="p1917202214387"></a>count</p>
</td>
<td class="cellrowborder" valign="top" width="11.93%" headers="mcps1.2.4.1.2 "><p id="p6917922183815"><a name="p6917922183815"></a><a name="p6917922183815"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.58%" headers="mcps1.2.4.1.3 "><p id="p29181822173818"><a name="p29181822173818"></a><a name="p29181822173818"></a>参与搬运的元素个数。</p>
</td>
</tr>
<tr id="row16554713131317"><td class="cellrowborder" valign="top" width="16.49%" headers="mcps1.2.4.1.1 "><p id="p1728791441620"><a name="p1728791441620"></a><a name="p1728791441620"></a>mask/mask[]</p>
</td>
<td class="cellrowborder" valign="top" width="11.93%" headers="mcps1.2.4.1.2 "><p id="p755431341319"><a name="p755431341319"></a><a name="p755431341319"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.58%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000002523303824_p0554313181312"><a name="zh-cn_topic_0000002523303824_p0554313181312"></a><a name="zh-cn_topic_0000002523303824_p0554313181312"></a><span id="zh-cn_topic_0000002523303824_ph793119540147"><a name="zh-cn_topic_0000002523303824_ph793119540147"></a><a name="zh-cn_topic_0000002523303824_ph793119540147"></a><a href="高维切分API.md#zh-cn_topic_0000002267504656_zh-cn_topic_0000001764162593_section4252658182">mask</a>用于控制每次迭代内参与计算的元素。</span></p>
<a name="zh-cn_topic_0000002523303824_ul1255411133132"></a><a name="zh-cn_topic_0000002523303824_ul1255411133132"></a><ul id="zh-cn_topic_0000002523303824_ul1255411133132"><li>逐bit模式：可以按位控制哪些元素参与计算，bit位的值为1表示参与计算，0表示不参与。<p id="zh-cn_topic_0000002523303824_p121114581013"><a name="zh-cn_topic_0000002523303824_p121114581013"></a><a name="zh-cn_topic_0000002523303824_p121114581013"></a>mask为数组形式，数组长度和数组元素的取值范围和操作数的数据类型有关。当操作数为16位时，数组长度为2，mask[0]、mask[1]∈[0, 2<sup id="zh-cn_topic_0000002523303824_sup1411059101"><a name="zh-cn_topic_0000002523303824_sup1411059101"></a><a name="zh-cn_topic_0000002523303824_sup1411059101"></a>64</sup>-1]并且不同时为0；当操作数为32位时，数组长度为1，mask[0]∈(0, 2<sup id="zh-cn_topic_0000002523303824_sup1711155161017"><a name="zh-cn_topic_0000002523303824_sup1711155161017"></a><a name="zh-cn_topic_0000002523303824_sup1711155161017"></a>64</sup>-1]；当操作数为64位时，数组长度为1，mask[0]∈(0, 2<sup id="zh-cn_topic_0000002523303824_sup181195111019"><a name="zh-cn_topic_0000002523303824_sup181195111019"></a><a name="zh-cn_topic_0000002523303824_sup181195111019"></a>32</sup>-1]。</p>
<p id="zh-cn_topic_0000002523303824_p711354105"><a name="zh-cn_topic_0000002523303824_p711354105"></a><a name="zh-cn_topic_0000002523303824_p711354105"></a>例如，mask=[8, 0]，8=0b1000，表示仅第4个元素参与计算。</p>
</li></ul>
<a name="zh-cn_topic_0000002523303824_ul18554121313135"></a><a name="zh-cn_topic_0000002523303824_ul18554121313135"></a><ul id="zh-cn_topic_0000002523303824_ul18554121313135"><li>连续模式：表示前面连续的多少个元素参与计算。取值范围和操作数的数据类型有关，数据类型不同，每次迭代内能够处理的元素个数最大值不同。当操作数为16位时，mask∈[1, 128]；当操作数为32位时，mask∈[1, 64]；当操作数为64位时，mask∈[1, 32]。</li></ul>
</td>
</tr>
<tr id="row185542138131"><td class="cellrowborder" valign="top" width="16.49%" headers="mcps1.2.4.1.1 "><p id="p755471321311"><a name="p755471321311"></a><a name="p755471321311"></a>repeatTime</p>
</td>
<td class="cellrowborder" valign="top" width="11.93%" headers="mcps1.2.4.1.2 "><p id="p135541313101314"><a name="p135541313101314"></a><a name="p135541313101314"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.58%" headers="mcps1.2.4.1.3 "><p id="p51982261556"><a name="p51982261556"></a><a name="p51982261556"></a>重复迭代次数。矢量计算单元，每次读取连续的256Bytes数据进行计算，为完成对输入数据的处理，必须通过多次迭代（repeat）才能完成所有数据的读取与计算。repeatTime表示迭代的次数。</p>
<p id="p17845145813432"><a name="p17845145813432"></a><a name="p17845145813432"></a>关于该参数的具体描述请参考<a href="高维切分API.md">高维切分API</a>。</p>
</td>
</tr>
<tr id="row195541813181310"><td class="cellrowborder" valign="top" width="16.49%" headers="mcps1.2.4.1.1 "><p id="p15554121320132"><a name="p15554121320132"></a><a name="p15554121320132"></a>repeatParams</p>
</td>
<td class="cellrowborder" valign="top" width="11.93%" headers="mcps1.2.4.1.2 "><p id="p18554141331317"><a name="p18554141331317"></a><a name="p18554141331317"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.58%" headers="mcps1.2.4.1.3 "><p id="p186307121111"><a name="p186307121111"></a><a name="p186307121111"></a>控制操作数地址步长的数据结构。CopyRepeatParams类型。</p>
<p id="p395104375712"><a name="p395104375712"></a><a name="p395104375712"></a>具体定义请参考<span id="ph10562197165916"><a name="ph10562197165916"></a><a name="ph10562197165916"></a>${INSTALL_DIR}</span>/include/ascendc/basic_api/interface/kernel_struct_data_copy.h，<span id="ph14322531015"><a name="ph14322531015"></a><a name="ph14322531015"></a>${INSTALL_DIR}</span>请替换为CANN软件安装后文件存储路径。</p>
<p id="p2065322712418"><a name="p2065322712418"></a><a name="p2065322712418"></a>参数说明请参考<a href="#table1940815635619">表3</a>。</p>
</td>
</tr>
</tbody>
</table>

**表 3**  CopyRepeatParams结构体参数说明

<a name="table1940815635619"></a>
<table><thead align="left"><tr id="row1940813563564"><th class="cellrowborder" valign="top" width="16.689999999999998%" id="mcps1.2.3.1.1"><p id="p1408155635620"><a name="p1408155635620"></a><a name="p1408155635620"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="83.31%" id="mcps1.2.3.1.2"><p id="p0409115655616"><a name="p0409115655616"></a><a name="p0409115655616"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row195147113498"><td class="cellrowborder" valign="top" width="16.689999999999998%" headers="mcps1.2.3.1.1 "><p id="p2037512181495"><a name="p2037512181495"></a><a name="p2037512181495"></a>dstStride、srcStride</p>
</td>
<td class="cellrowborder" valign="top" width="83.31%" headers="mcps1.2.3.1.2 "><p id="p14375151815499"><a name="p14375151815499"></a><a name="p14375151815499"></a>用于设置同一迭代内datablock的地址步长，取值范围为[0,65535]。</p>
<p id="p18375201854911"><a name="p18375201854911"></a><a name="p18375201854911"></a>同一迭代内datablock的地址步长参数说明请参考<a href="高维切分API.md#zh-cn_topic_0000002267504656_zh-cn_topic_0000001764162593_section2815124173416">dataBlockStride</a>。</p>
</td>
</tr>
<tr id="row340910561569"><td class="cellrowborder" valign="top" width="16.689999999999998%" headers="mcps1.2.3.1.1 "><p id="p1340945612566"><a name="p1340945612566"></a><a name="p1340945612566"></a>dstRepeatSize、srcRepeatSize</p>
</td>
<td class="cellrowborder" valign="top" width="83.31%" headers="mcps1.2.3.1.2 "><p id="p204741950721"><a name="p204741950721"></a><a name="p204741950721"></a>用于设置相邻迭代间的地址步长，取值范围为[0,4095]。</p>
<p id="p18391947266"><a name="p18391947266"></a><a name="p18391947266"></a>相邻迭代间的地址步长参数说明请参考<a href="高维切分API.md#zh-cn_topic_0000002267504656_zh-cn_topic_0000001764162593_section139459347420">repeatStride</a>。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

-   源操作数和目的操作数的起始地址需要保证32字节对齐。
-   tensor前n个数据计算接口仅支持Ascend 950PR/Ascend 950DT。
-   针对Ascend 950PR/Ascend 950DT，uint8\_t/int8\_t/hifloat8\_t/fp8\_e4m3fn\_t/fp8\_e5m2\_t/fp4x2\_e2m1\_t/fp4x2\_e1m2\_t/fp8\_e8m0\_t/uint64\_t/int64\_t数据类型仅支持tensor前n个数据计算接口。
-   针对Ascend 950PR/Ascend 950DT，tensor前n个数据计算接口中的isSetMask参数不生效，保持默认值即可。
-   Copy和矢量计算API一样，支持和掩码操作API配合使用。但Counter模式配合高维切分计算API时，和[通用的Counter模式](如何使用掩码操作API.md)有一定差异。具体差异如下：
    -   通用的Counter模式：Mask代表**整个矢量计算参与计算的元素个数，迭代次数不生效**。
    -   Counter模式配合Copy高维切分计算API，Mask代表**每次Repeat中处理的元素个数，迭代次数生效。**示意图如下：

        <!-- img2text -->
```text
srcLocal
<────────────────────────────────────────────────────────────>
                      srcRepeatSize
<────────>
 srcStride

┌──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┐
│  1   │      │  2   │      │  3   │  …   │  N   │  …   │  1   │      │  2   │      │  3   │  …   │
└──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┘
                                          N=ceil(Mask*sizeof(T)/datablockSize)

<──────>
  bk1
        <──────>
          bk2
                  <──────>
                    bk3
                                    <──────>
                                      bkN
<────────────────────────────────────────────>
                   repeat 1

                                                            <──────>
                                                              bk1
                                                                      <──────>
                                                                        bk2
                                                                                      <──────>
                                                                                        bk3
                                                                                                            <──────>
                                                                                                              bkN
                                                            <────────────────────────────────────────────>
                                                                                   repeat 2


dstLocal
<────────────────────────────────────────────────────>
                   dstRepeatSize
<──────>
dstStride

┌──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┐
│  1   │      │  2   │      │  3   │  …   │  N   │  …   │  1   │      │  2   │  3   │  …   │
└──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┘

<──────>
  bk1
        <──────>
          bk2
                  <──────>
                    bk3
                                    <──────>
                                      bkN
<────────────────────────────────────>
               repeat 1

                                                      <──────>
                                                        bk1
                                                              <──────>
                                                                bk2
                                                                        <──────>
                                                                          bk3
                                                                                          <──────>
                                                                                            bkN
                                                      <────────────────────────────────────>
                                                                     repeat 2
```

说明:
- 图中 srcLocal 与 dstLocal 都展示了按 repeat 分组的数据布局。
- `N=ceil(Mask*sizeof(T)/datablockSize)` 表示每次 Repeat 中处理的数据块个数 N。
- `srcRepeatSize` / `dstRepeatSize` 表示相邻两次 repeat 起始位置之间的跨度。
- `srcStride` / `dstStride` 表示同一次 repeat 内，相邻数据块 `bk1、bk2、bk3 ... bkN` 之间的间隔。
- `repeat 1`、`repeat 2` 表示迭代次数生效；每次 repeat 只处理由 Mask 决定的元素数量。
- `bk1`、`bk2`、`bk3`、`bkN` 为每次 repeat 内的各个 block。

## 调用示例<a name="section1227835243314"></a>

本示例仅展示Compute流程中的部分代码。如需运行，请参考[Copy样例](https://gitcode.com/cann/asc-devkit/tree/master/examples/02_features/07_data_movement/copy)实现完整的代码。

本示例中操作数数据类型为int16\_t。

-   tensor前n个数据计算

    ```
    AscendC::Copy(dstLocal, srcLocal, 512);
    ```

    结果示例如下：

    ```
    输入数据srcLocal：[9 -2 8 ... 9]
    输出数据dstLocal:
    [9 -2 8 ... 9]
    ```

-   mask连续模式

    ```
    uint64_t mask = 128;
    // repeatTime = 4, 128 elements one repeat, 512 elements total
    // dstStride, srcStride = 1, no gap between blocks in one repeat
    // dstRepStride, srcRepStride = 8, no gap between repeats
    AscendC::Copy(dstLocal, srcLocal, mask, 4, { 1, 1, 8, 8 });
    ```

    结果示例如下：

    ```
    输入数据srcLocal:[9 -2 8 ... 9]
    输出数据dstLocal:
    [9 -2 8 ... 9]
    ```

-   mask逐bit模式

    ```
    uint64_t mask[2] = { UINT64_MAX, UINT64_MAX };
    // repeatTime = 4, 128 elements one repeat, 512 elements total
    // dstStride, srcStride = 1, no gap between blocks in one repeat
    // dstRepStride, srcRepStride = 8, no gap between repeats
    AscendC::Copy(dstLocal, srcLocal, mask, 4, { 1, 1, 8, 8 });
    ```

    结果示例如下：

    ```
    输入数据srcLocal：[9 -2 8 ... 9]
    输出数据dstLocal:
    [9 -2 8 ... 9]
    ```

