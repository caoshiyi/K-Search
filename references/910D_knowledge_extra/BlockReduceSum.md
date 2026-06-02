# BlockReduceSum<a name="ZH-CN_TOPIC_0000002523304192"></a>

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

对每个datablock内所有元素求和。源操作数相加采用二叉树方式，两两相加。归约指令的总体介绍请参考[如何使用归约计算API](如何使用归约计算API.md)。

以128个half类型的数据求和为例，每个datablock可以计算16个half类型数据，分成8个datablock进行计算；每个datablock内，通过二叉树的方式，两两相加，BlockReduceSum求和示意图如下。

**图 1**  BlockReduceSum求和示意图<a name="fig428072895917"></a>  
<!-- img2text -->
```
datablock0
datablock1
...
datablock7
      ─────→

┌──────────────────────────────────────────────────────────────────────────────────────────────┐
│  ┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐                                          │
│  │0 │1 │2 │3 │4 │5 │6 │7 │8 │9 │10│11│12│13│14│15│                                          │
│  └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘                                          │
│    │  │   │  │   │  │   │  │   │  │   │  │   │  │   │  │                                   │
│    └┬─┘   └┬─┘   └┬─┘   └┬─┘   └┬─┘   └┬─┘   └┬─┘   └┬─┘                                   │
│   ┌────┐  ┌────┐  ┌────┐  ┌────┐  ┌────┐  ┌────┐  ┌────┐  ┌────┐                            │
│   │ 00 │  │ 01 │  │ 02 │  │ 03 │  │ 04 │  │ 05 │  │ 06 │  │ 07 │                            │
│   └────┘  └────┘  └────┘  └────┘  └────┘  └────┘  └────┘  └────┘                            │
│      ╲      ╱        ╲      ╱        ╲      ╱        ╲      ╱                               │
│       ╲    ╱          ╲    ╱          ╲    ╱          ╲    ╱                                │
│      ┌─────┐         ┌─────┐         ┌─────┐         ┌─────┐                                │
│      │ 000 │         │ 001 │         │ 002 │         │ 003 │                                │
│      └─────┘         └─────┘         └─────┘         └─────┘                                │
│          ╲              ╱               ╲              ╱                                     │
│           ╲            ╱                 ╲            ╱                                      │
│            ╲          ╱                   ╲          ╱                                       │
│             ┌──────┐                         ┌──────┐                                       │
│             │ 0000 │                         │ 0001 │                                       │
│             └──────┘                         └──────┘                                       │
│                  ╲                             ╱                                            │
│                   ╲                           ╱                                             │
│                    ╲                         ╱                                              │
│                     ╲                       ╱                                               │
│                      ╲                     ╱                                                │
│                       ┌───────────┐                                                        │
│                       │datablockn │                                                        │
│                       └───────────┘                                                        │
└──────────────────────────────────────────────────────────────────────────────────────────────┘
```

需要注意的是两两相加的计算过程中，计算结果大于65504时结果保存为65504。例如，源操作数为\[60000,60000,-30000,100\]，首先60000+60000溢出，结果为65504，然后计算-30000+100=-29900，最后计算65504-29900=35604，计算示意图如下图所示。

**图 2**  存在溢出场景时的计算示意图<a name="fig8394103233418"></a>  
<!-- img2text -->
```
┌────────┐      ┌────────┐      ┌─────────┐      ┌─────┐
│ 60000  │      │ 60000  │      │ -30000  │      │ 100 │
└────────┘      └────────┘      └─────────┘      └─────┘
      \            /                    \            /
       \          /                      \          /
        \        /                        \        /
         ▼      ▼                          ▼      ▼
       溢出  ┌────────┐                ┌─────────┐
             │ 65504  │                │ -29900  │
             └────────┘                └─────────┘
                  \                        /
                   \                      /
                    \                    /
                     ▼                  ▼
                     ┌────────┐
                     │ 35604  │
                     └────────┘
```

## 函数原型<a name="section620mcpsimp"></a>

-   mask逐比特模式

    ```
    template <typename T, bool isSetMask = true>
    __aicore__ inline void BlockReduceSum(const LocalTensor<T>& dst, const LocalTensor<T>& src,const int32_t repeatTime, const uint64_t mask[], const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)
    ```

-   mask连续模式

    ```
    template <typename T, bool isSetMask = true>
    __aicore__ inline void BlockReduceSum(const LocalTensor<T>& dst, const LocalTensor<T>& src,const int32_t repeatTime, const int32_t mask, const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)
    ```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table4835205712588"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001429830437_row118356578583"><th class="cellrowborder" valign="top" width="13.52%" id="mcps1.2.3.1.1"><p id="zh-cn_topic_0000001429830437_p48354572582"><a name="zh-cn_topic_0000001429830437_p48354572582"></a><a name="zh-cn_topic_0000001429830437_p48354572582"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="86.48%" id="mcps1.2.3.1.2"><p id="zh-cn_topic_0000001429830437_p583535795817"><a name="zh-cn_topic_0000001429830437_p583535795817"></a><a name="zh-cn_topic_0000001429830437_p583535795817"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001429830437_row1835857145817"><td class="cellrowborder" valign="top" width="13.52%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001429830437_p5835457165816"><a name="zh-cn_topic_0000001429830437_p5835457165816"></a><a name="zh-cn_topic_0000001429830437_p5835457165816"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="86.48%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001429830437_p168351657155818"><a name="zh-cn_topic_0000001429830437_p168351657155818"></a><a name="zh-cn_topic_0000001429830437_p168351657155818"></a>操作数数据类型。</p>
<p id="p13029151498"><a name="p13029151498"></a><a name="p13029151498"></a><span id="ph19302171510497"><a name="ph19302171510497"></a><a name="ph19302171510497"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：half、float</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001429830437_row18835145716587"><td class="cellrowborder" valign="top" width="13.52%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001429830437_p1383515717581"><a name="zh-cn_topic_0000001429830437_p1383515717581"></a><a name="zh-cn_topic_0000001429830437_p1383515717581"></a>isSetMask</p>
</td>
<td class="cellrowborder" valign="top" width="86.48%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002554343931_p77520541653"><a name="zh-cn_topic_0000002554343931_p77520541653"></a><a name="zh-cn_topic_0000002554343931_p77520541653"></a>是否在接口内部设置mask。</p>
<a name="zh-cn_topic_0000002554343931_ul1163765616511"></a><a name="zh-cn_topic_0000002554343931_ul1163765616511"></a><ul id="zh-cn_topic_0000002554343931_ul1163765616511"><li>true，表示在接口内部设置mask。</li><li>false，表示在接口外部设置mask，开发者需要使用<a href="SetVectorMask.md">SetVectorMask</a>接口设置mask值。这种模式下，本接口入参中的mask值必须设置为占位符MASK_PLACEHOLDER。</li></ul>
</td>
</tr>
</tbody>
</table>

**表 2**  参数说明

<a name="table8955841508"></a>
<table><thead align="left"><tr id="row15956194105014"><th class="cellrowborder" valign="top" width="13.661366136613662%" id="mcps1.2.4.1.1"><p id="p7956144195014"><a name="p7956144195014"></a><a name="p7956144195014"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="12.591259125912593%" id="mcps1.2.4.1.2"><p id="p1295624145013"><a name="p1295624145013"></a><a name="p1295624145013"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="73.74737473747375%" id="mcps1.2.4.1.3"><p id="p16956144145011"><a name="p16956144145011"></a><a name="p16956144145011"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row5956546509"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p284425844311"><a name="p284425844311"></a><a name="p284425844311"></a>dst</p>
</td>
<td class="cellrowborder" valign="top" width="12.591259125912593%" headers="mcps1.2.4.1.2 "><p id="p158449584436"><a name="p158449584436"></a><a name="p158449584436"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="73.74737473747375%" headers="mcps1.2.4.1.3 "><p id="p1297092014113"><a name="p1297092014113"></a><a name="p1297092014113"></a>目的操作数。</p>
<p id="p16703131355116"><a name="p16703131355116"></a><a name="p16703131355116"></a><span id="zh-cn_topic_0000002523303824_ph173308471594"><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><span id="zh-cn_topic_0000002523303824_ph9902231466"><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816"><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p3820115971115"><a name="p3820115971115"></a><a name="p3820115971115"></a>LocalTensor的起始地址需要保证16字节对齐（针对half数据类型），32字节对齐（针对float数据类型）。</p>
</td>
</tr>
<tr id="row4956154125018"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p142871414131614"><a name="p142871414131614"></a><a name="p142871414131614"></a>src</p>
</td>
<td class="cellrowborder" valign="top" width="12.591259125912593%" headers="mcps1.2.4.1.2 "><p id="p628711148165"><a name="p628711148165"></a><a name="p628711148165"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.74737473747375%" headers="mcps1.2.4.1.3 "><p id="p1388703910412"><a name="p1388703910412"></a><a name="p1388703910412"></a>源操作数。</p>
<p id="p14260114212414"><a name="p14260114212414"></a><a name="p14260114212414"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_1"><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_1"><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_1"><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p78204190123"><a name="p78204190123"></a><a name="p78204190123"></a><span id="ph1479701815419"><a name="ph1479701815419"></a><a name="ph1479701815419"></a>LocalTensor的起始地址需要32字节对齐。</span></p>
</td>
</tr>
<tr id="row1495634115010"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p028761412166"><a name="p028761412166"></a><a name="p028761412166"></a>repeatTime</p>
</td>
<td class="cellrowborder" valign="top" width="12.591259125912593%" headers="mcps1.2.4.1.2 "><p id="p1928716142163"><a name="p1928716142163"></a><a name="p1928716142163"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.74737473747375%" headers="mcps1.2.4.1.3 "><p id="p353564621520"><a name="p353564621520"></a><a name="p353564621520"></a>迭代次数。取值范围为[0, 255]。</p>
<p id="p17845145813432"><a name="p17845145813432"></a><a name="p17845145813432"></a>关于该参数的具体描述请参考<a href="高维切分API.md">高维切分API</a>。</p>
</td>
</tr>
<tr id="row1075785651510"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p1728791441620"><a name="p1728791441620"></a><a name="p1728791441620"></a>mask/mask[]</p>
</td>
<td class="cellrowborder" valign="top" width="12.591259125912593%" headers="mcps1.2.4.1.2 "><p id="p11287151451610"><a name="p11287151451610"></a><a name="p11287151451610"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.74737473747375%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000002523303824_p0554313181312"><a name="zh-cn_topic_0000002523303824_p0554313181312"></a><a name="zh-cn_topic_0000002523303824_p0554313181312"></a><span id="zh-cn_topic_0000002523303824_ph793119540147"><a name="zh-cn_topic_0000002523303824_ph793119540147"></a><a name="zh-cn_topic_0000002523303824_ph793119540147"></a><a href="高维切分API.md#zh-cn_topic_0000002267504656_zh-cn_topic_0000001764162593_section4252658182">mask</a>用于控制每次迭代内参与计算的元素。</span></p>
<a name="zh-cn_topic_0000002523303824_ul1255411133132"></a><a name="zh-cn_topic_0000002523303824_ul1255411133132"></a><ul id="zh-cn_topic_0000002523303824_ul1255411133132"><li>逐bit模式：可以按位控制哪些元素参与计算，bit位的值为1表示参与计算，0表示不参与。<p id="zh-cn_topic_0000002523303824_p121114581013"><a name="zh-cn_topic_0000002523303824_p121114581013"></a><a name="zh-cn_topic_0000002523303824_p121114581013"></a>mask为数组形式，数组长度和数组元素的取值范围和操作数的数据类型有关。当操作数为16位时，数组长度为2，mask[0]、mask[1]∈[0, 2<sup id="zh-cn_topic_0000002523303824_sup1411059101"><a name="zh-cn_topic_0000002523303824_sup1411059101"></a><a name="zh-cn_topic_0000002523303824_sup1411059101"></a>64</sup>-1]并且不同时为0；当操作数为32位时，数组长度为1，mask[0]∈(0, 2<sup id="zh-cn_topic_0000002523303824_sup1711155161017"><a name="zh-cn_topic_0000002523303824_sup1711155161017"></a><a name="zh-cn_topic_0000002523303824_sup1711155161017"></a>64</sup>-1]；当操作数为64位时，数组长度为1，mask[0]∈(0, 2<sup id="zh-cn_topic_0000002523303824_sup181195111019"><a name="zh-cn_topic_0000002523303824_sup181195111019"></a><a name="zh-cn_topic_0000002523303824_sup181195111019"></a>32</sup>-1]。</p>
<p id="zh-cn_topic_0000002523303824_p711354105"><a name="zh-cn_topic_0000002523303824_p711354105"></a><a name="zh-cn_topic_0000002523303824_p711354105"></a>例如，mask=[8, 0]，8=0b1000，表示仅第4个元素参与计算。</p>
</li></ul>
<a name="zh-cn_topic_0000002523303824_ul18554121313135"></a><a name="zh-cn_topic_0000002523303824_ul18554121313135"></a><ul id="zh-cn_topic_0000002523303824_ul18554121313135"><li>连续模式：表示前面连续的多少个元素参与计算。取值范围和操作数的数据类型有关，数据类型不同，每次迭代内能够处理的元素个数最大值不同。当操作数为16位时，mask∈[1, 128]；当操作数为32位时，mask∈[1, 64]；当操作数为64位时，mask∈[1, 32]。</li></ul>
</td>
</tr>
<tr id="row10734115313202"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p19575838172212"><a name="p19575838172212"></a><a name="p19575838172212"></a>dstRepStride</p>
</td>
<td class="cellrowborder" valign="top" width="12.591259125912593%" headers="mcps1.2.4.1.2 "><p id="p35752385220"><a name="p35752385220"></a><a name="p35752385220"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.74737473747375%" headers="mcps1.2.4.1.3 "><p id="p9558101519355"><a name="p9558101519355"></a><a name="p9558101519355"></a>目的操作数相邻迭代间的地址步长。以一个repeatTime归约后的长度为单位。</p>
<p id="p45751038192213"><a name="p45751038192213"></a><a name="p45751038192213"></a>每个repeatTime(8个datablock)归约后，得到8个元素，所以输入类型为half类型时，RepStride单位为16Byte；输入类型为float类型时，RepStride单位为32Byte。</p>
</td>
</tr>
<tr id="row548531101613"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p62871914191620"><a name="p62871914191620"></a><a name="p62871914191620"></a>srcBlkStride</p>
</td>
<td class="cellrowborder" valign="top" width="12.591259125912593%" headers="mcps1.2.4.1.2 "><p id="p5287181416164"><a name="p5287181416164"></a><a name="p5287181416164"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.74737473747375%" headers="mcps1.2.4.1.3 "><p id="p14215346174119"><a name="p14215346174119"></a><a name="p14215346174119"></a>单次迭代内datablock的地址步长。详细说明请参考<a href="高维切分API.md#zh-cn_topic_0000002267504656_zh-cn_topic_0000001764162593_section2815124173416">dataBlockStride</a>。</p>
</td>
</tr>
<tr id="row1774899165"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p15288514111614"><a name="p15288514111614"></a><a name="p15288514111614"></a>srcRepStride</p>
</td>
<td class="cellrowborder" valign="top" width="12.591259125912593%" headers="mcps1.2.4.1.2 "><p id="p62888148164"><a name="p62888148164"></a><a name="p62888148164"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.74737473747375%" headers="mcps1.2.4.1.3 "><p id="p1624214011488"><a name="p1624214011488"></a><a name="p1624214011488"></a>源操作数相邻迭代间的地址步长，即源操作数每次迭代跳过的datablock数目。详细说明请参考<a href="高维切分API.md#zh-cn_topic_0000002267504656_zh-cn_topic_0000001764162593_section139459347420">repeatStride</a>。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section128671456102513"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

-   操作数地址对齐要求请参见[通用地址对齐约束](通用说明和约束.md#section796754519912)。

-   为了节省地址空间，您可以定义一个Tensor，供源操作数与目的操作数同时使用（即地址重叠），需要注意计算后的目的操作数数据不能覆盖未参与计算的源操作数，需要谨慎使用。

## 调用示例<a name="section642mcpsimp"></a>

-   本样例中只展示Compute流程中的部分代码。完整样例可参考[block\_reduce\_max样例](https://gitcode.com/cann/asc-devkit/tree/master/examples/03_libraries/05_reduce/block_reduce_max)，并将下列代码片段拷贝并替换完整样例中的相关代码。

    -   BlockReduceSum-tensor高维切分计算样例-mask连续模式

        ```
        // 设定mask为最多的128个全部元素参与计算
        int32_t mask = 256/sizeof(half);
        // 每个repeat128个元素，一共128个元素。
        int repeat = 1;
        // dstLocal: 目的操作数tensor
        // srcLocal: 源操作数tensor
        // srcBlkStride = 1, 在一个repeat中，block间没有空隙。
        // dstRepStride = 1, srcRepStride = 8, repeat间没有空隙。
        AscendC::BlockReduceSum<half>(dstLocal, srcLocal, repeat, mask, 1, 1, 8);
        ```

    -   BlockReduceSum-tensor高维切分计算样例-mask逐bit模式

        ```
        // 设定mask为最多的128个全部元素参与计算
        uint64_t mask[2] = { UINT64_MAX, UINT64_MAX };
        // 每个repeat128个元素，一共128个元素。
        int repeat = 1;
        // dstLocal: 目的操作数tensor
        // srcLocal: 源操作数tensor
        // srcBlkStride = 1, 在一个repeat中，block间没有空隙。
        // dstRepStride = 1, srcRepStride = 8, repeat间没有空隙。
        AscendC::BlockReduceSum<half>(dstLocal, srcLocal, repeat, mask, 1, 1, 8);
        ```

    结果示例如下：

    ```
    输入数据src_gm: 
    [1, 1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1, 1,
     2, 2, 2, 2, 2, 2, 2, 2,
     2, 2, 2, 2, 2, 2, 2, 2,
     ... 
     3, 3, 3, 3, 3, 3, 3, 3,
     3, 3, 3, 3, 3, 3, 3, 3]
    
    输出数据dst_gm: 
    [16, 32, ..., 48]
    ```

    ```
    输入数据src_gm：
    [-7.289, 4.48, -5.898, -6.199, 1.422, -6.168, -3.178, -1.198, 
     7.789, 6.754, -5.191, -0.6797, 2.883, 2.08, 8.664, -8.539,
     ...,
     -7.625, 2.529, 7.855, -2.012, -6.52, -6.652, -8.422, -9.914,
     -4.355, 1.849, 5.406, 1.483, -6.074, -1.897, 8.625, 1.969]  
    输出数据dst_gm：
    [-10.27, ..., -23.77, 0, ..., 0]
    ```

-   针对不同场景合理使用归约指令可以带来性能提升，相关介绍请参考[选择低延迟指令，优化归约操作性能](选择低延迟指令-优化归约操作性能.md)，具体样例请参考[ReduceCustom](https://gitee.com/ascend/samples/tree/master/operator/ascendc/0_introduction/14_reduce_frameworklaunch/ReduceCustom)。

