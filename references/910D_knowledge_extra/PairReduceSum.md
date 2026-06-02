# PairReduceSum<a name="ZH-CN_TOPIC_0000002554344263"></a>

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

PairReduceSum：相邻两个（奇偶）元素求和，例如（a1, a2, a3, a4, a5, a6...），相邻两个数据求和为（a1+a2,  a3+a4,  a5+a6, ......）。归约指令的总体介绍请参考[如何使用归约计算API](如何使用归约计算API.md)。

## 函数原型<a name="section620mcpsimp"></a>

-   mask逐bit模式

    ```
    template <typename T, bool isSetMask = true>
    __aicore__ inline void PairReduceSum(const LocalTensor<T>& dst, const LocalTensor<T>& src, const int32_t repeatTime, const uint64_t mask[], const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)
    ```

-   mask连续模式

    ```
    template <typename T, bool isSetMask = true>
    __aicore__ inline void PairReduceSum(const LocalTensor<T>& dst, const LocalTensor<T>& src, const int32_t repeatTime, const int32_t mask, const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)
    ```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table4835205712588"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001429830437_row118356578583"><th class="cellrowborder" valign="top" width="13.420000000000002%" id="mcps1.2.3.1.1"><p id="zh-cn_topic_0000001429830437_p48354572582"><a name="zh-cn_topic_0000001429830437_p48354572582"></a><a name="zh-cn_topic_0000001429830437_p48354572582"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="86.58%" id="mcps1.2.3.1.2"><p id="zh-cn_topic_0000001429830437_p583535795817"><a name="zh-cn_topic_0000001429830437_p583535795817"></a><a name="zh-cn_topic_0000001429830437_p583535795817"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001429830437_row1835857145817"><td class="cellrowborder" valign="top" width="13.420000000000002%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001429830437_p5835457165816"><a name="zh-cn_topic_0000001429830437_p5835457165816"></a><a name="zh-cn_topic_0000001429830437_p5835457165816"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="86.58%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001429830437_p168351657155818"><a name="zh-cn_topic_0000001429830437_p168351657155818"></a><a name="zh-cn_topic_0000001429830437_p168351657155818"></a>操作数数据类型。</p>
<p id="p778521092"><a name="p778521092"></a><a name="p778521092"></a><span id="ph133209151609"><a name="ph133209151609"></a><a name="ph133209151609"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：half/float</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001429830437_row18835145716587"><td class="cellrowborder" valign="top" width="13.420000000000002%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001429830437_p1383515717581"><a name="zh-cn_topic_0000001429830437_p1383515717581"></a><a name="zh-cn_topic_0000001429830437_p1383515717581"></a>isSetMask</p>
</td>
<td class="cellrowborder" valign="top" width="86.58%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002554343931_p77520541653"><a name="zh-cn_topic_0000002554343931_p77520541653"></a><a name="zh-cn_topic_0000002554343931_p77520541653"></a>是否在接口内部设置mask。</p>
<a name="zh-cn_topic_0000002554343931_ul1163765616511"></a><a name="zh-cn_topic_0000002554343931_ul1163765616511"></a><ul id="zh-cn_topic_0000002554343931_ul1163765616511"><li>true，表示在接口内部设置mask。</li><li>false，表示在接口外部设置mask，开发者需要使用<a href="SetVectorMask.md">SetVectorMask</a>接口设置mask值。这种模式下，本接口入参中的mask值必须设置为占位符MASK_PLACEHOLDER。</li></ul>
</td>
</tr>
</tbody>
</table>

**表 2**  参数说明

<a name="table8955841508"></a>
<table><thead align="left"><tr id="row15956194105014"><th class="cellrowborder" valign="top" width="13.661366136613662%" id="mcps1.2.4.1.1"><p id="p7956144195014"><a name="p7956144195014"></a><a name="p7956144195014"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="12.58125812581258%" id="mcps1.2.4.1.2"><p id="p1295624145013"><a name="p1295624145013"></a><a name="p1295624145013"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="73.75737573757377%" id="mcps1.2.4.1.3"><p id="p16956144145011"><a name="p16956144145011"></a><a name="p16956144145011"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row5956546509"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p284425844311"><a name="p284425844311"></a><a name="p284425844311"></a>dst</p>
</td>
<td class="cellrowborder" valign="top" width="12.58125812581258%" headers="mcps1.2.4.1.2 "><p id="p158449584436"><a name="p158449584436"></a><a name="p158449584436"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="73.75737573757377%" headers="mcps1.2.4.1.3 "><p id="p17321554104118"><a name="p17321554104118"></a><a name="p17321554104118"></a>目的操作数。</p>
<p id="p16703131355116"><a name="p16703131355116"></a><a name="p16703131355116"></a><span id="zh-cn_topic_0000002523303824_ph173308471594"><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><span id="zh-cn_topic_0000002523303824_ph9902231466"><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816"><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p78204190123"><a name="p78204190123"></a><a name="p78204190123"></a><span id="ph1479701815419"><a name="ph1479701815419"></a><a name="ph1479701815419"></a>LocalTensor的起始地址需要32字节对齐。</span></p>
</td>
</tr>
<tr id="row4956154125018"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p142871414131614"><a name="p142871414131614"></a><a name="p142871414131614"></a>src</p>
</td>
<td class="cellrowborder" valign="top" width="12.58125812581258%" headers="mcps1.2.4.1.2 "><p id="p628711148165"><a name="p628711148165"></a><a name="p628711148165"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.75737573757377%" headers="mcps1.2.4.1.3 "><p id="p1197411244216"><a name="p1197411244216"></a><a name="p1197411244216"></a>源操作数。</p>
<p id="p1225766114218"><a name="p1225766114218"></a><a name="p1225766114218"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_1"><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_1"><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_1"><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p194546161418"><a name="p194546161418"></a><a name="p194546161418"></a><span id="ph441646171420"><a name="ph441646171420"></a><a name="ph441646171420"></a>LocalTensor的起始地址需要32字节对齐。</span></p>
</td>
</tr>
<tr id="row1495634115010"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p028761412166"><a name="p028761412166"></a><a name="p028761412166"></a>repeatTime</p>
</td>
<td class="cellrowborder" valign="top" width="12.58125812581258%" headers="mcps1.2.4.1.2 "><p id="p1928716142163"><a name="p1928716142163"></a><a name="p1928716142163"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.75737573757377%" headers="mcps1.2.4.1.3 "><p id="p353564621520"><a name="p353564621520"></a><a name="p353564621520"></a>迭代次数。取值范围为[0, 255]。</p>
<p id="p216133820197"><a name="p216133820197"></a><a name="p216133820197"></a>关于该参数的具体描述请参考<a href="高维切分API.md">高维切分API</a>。</p>
</td>
</tr>
<tr id="row1075785651510"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p5535723122215"><a name="p5535723122215"></a><a name="p5535723122215"></a>mask/mask[]</p>
</td>
<td class="cellrowborder" valign="top" width="12.58125812581258%" headers="mcps1.2.4.1.2 "><p id="p11287151451610"><a name="p11287151451610"></a><a name="p11287151451610"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.75737573757377%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000002523303824_p0554313181312"><a name="zh-cn_topic_0000002523303824_p0554313181312"></a><a name="zh-cn_topic_0000002523303824_p0554313181312"></a><span id="zh-cn_topic_0000002523303824_ph793119540147"><a name="zh-cn_topic_0000002523303824_ph793119540147"></a><a name="zh-cn_topic_0000002523303824_ph793119540147"></a><a href="高维切分API.md#zh-cn_topic_0000002267504656_zh-cn_topic_0000001764162593_section4252658182">mask</a>用于控制每次迭代内参与计算的元素。</span></p>
<a name="zh-cn_topic_0000002523303824_ul1255411133132"></a><a name="zh-cn_topic_0000002523303824_ul1255411133132"></a><ul id="zh-cn_topic_0000002523303824_ul1255411133132"><li>逐bit模式：可以按位控制哪些元素参与计算，bit位的值为1表示参与计算，0表示不参与。<p id="zh-cn_topic_0000002523303824_p121114581013"><a name="zh-cn_topic_0000002523303824_p121114581013"></a><a name="zh-cn_topic_0000002523303824_p121114581013"></a>mask为数组形式，数组长度和数组元素的取值范围和操作数的数据类型有关。当操作数为16位时，数组长度为2，mask[0]、mask[1]∈[0, 2<sup id="zh-cn_topic_0000002523303824_sup1411059101"><a name="zh-cn_topic_0000002523303824_sup1411059101"></a><a name="zh-cn_topic_0000002523303824_sup1411059101"></a>64</sup>-1]并且不同时为0；当操作数为32位时，数组长度为1，mask[0]∈(0, 2<sup id="zh-cn_topic_0000002523303824_sup1711155161017"><a name="zh-cn_topic_0000002523303824_sup1711155161017"></a><a name="zh-cn_topic_0000002523303824_sup1711155161017"></a>64</sup>-1]；当操作数为64位时，数组长度为1，mask[0]∈(0, 2<sup id="zh-cn_topic_0000002523303824_sup181195111019"><a name="zh-cn_topic_0000002523303824_sup181195111019"></a><a name="zh-cn_topic_0000002523303824_sup181195111019"></a>32</sup>-1]。</p>
<p id="zh-cn_topic_0000002523303824_p711354105"><a name="zh-cn_topic_0000002523303824_p711354105"></a><a name="zh-cn_topic_0000002523303824_p711354105"></a>例如，mask=[8, 0]，8=0b1000，表示仅第4个元素参与计算。</p>
</li></ul>
<a name="zh-cn_topic_0000002523303824_ul18554121313135"></a><a name="zh-cn_topic_0000002523303824_ul18554121313135"></a><ul id="zh-cn_topic_0000002523303824_ul18554121313135"><li>连续模式：表示前面连续的多少个元素参与计算。取值范围和操作数的数据类型有关，数据类型不同，每次迭代内能够处理的元素个数最大值不同。当操作数为16位时，mask∈[1, 128]；当操作数为32位时，mask∈[1, 64]；当操作数为64位时，mask∈[1, 32]。</li></ul>
</td>
</tr>
<tr id="row47221542143511"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p192887146163"><a name="p192887146163"></a><a name="p192887146163"></a>dstRepStride</p>
</td>
<td class="cellrowborder" valign="top" width="12.58125812581258%" headers="mcps1.2.4.1.2 "><p id="p1628815142167"><a name="p1628815142167"></a><a name="p1628815142167"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.75737573757377%" headers="mcps1.2.4.1.3 "><p id="p9558101519355"><a name="p9558101519355"></a><a name="p9558101519355"></a>目的操作数相邻迭代间的地址步长。以一个repeat归约后的长度为单位。PairReduce完成后，一个repeat的长度减半。即单位为128Byte。</p>
</td>
</tr>
<tr id="row548531101613"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p62871914191620"><a name="p62871914191620"></a><a name="p62871914191620"></a>srcBlkStride</p>
</td>
<td class="cellrowborder" valign="top" width="12.58125812581258%" headers="mcps1.2.4.1.2 "><p id="p5287181416164"><a name="p5287181416164"></a><a name="p5287181416164"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.75737573757377%" headers="mcps1.2.4.1.3 "><p id="p14215346174119"><a name="p14215346174119"></a><a name="p14215346174119"></a>单次迭代内datablock的地址步长。详细说明请参考<a href="高维切分API.md#zh-cn_topic_0000002267504656_zh-cn_topic_0000001764162593_section2815124173416">dataBlockStride</a>。</p>
</td>
</tr>
<tr id="row1774899165"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p15288514111614"><a name="p15288514111614"></a><a name="p15288514111614"></a>srcRepStride</p>
</td>
<td class="cellrowborder" valign="top" width="12.58125812581258%" headers="mcps1.2.4.1.2 "><p id="p62888148164"><a name="p62888148164"></a><a name="p62888148164"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.75737573757377%" headers="mcps1.2.4.1.3 "><p id="p1624214011488"><a name="p1624214011488"></a><a name="p1624214011488"></a>源操作数相邻迭代间的地址步长，即源操作数每次迭代跳过的datablock数目。详细说明请参考<a href="高维切分API.md#zh-cn_topic_0000002267504656_zh-cn_topic_0000001764162593_section139459347420">repeatStride</a>。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section128671456102513"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

-   操作数地址对齐要求请参见[通用地址对齐约束](通用说明和约束.md#section796754519912)。

## 调用示例<a name="section642mcpsimp"></a>

本样例中只展示Compute流程中的部分代码。完整样例可参考[pair\_reduce\_sum样例](https://gitcode.com/cann/asc-devkit/tree/master/examples/03_libraries/05_reduce/pair_reduce_sum)

-   PairReduceSum-tensor高维切分计算样例-mask连续模式

    ```
    // 设定mask为最多的128个全部元素参与计算
    int32_t mask = 256/sizeof(half);
    // 每个repeat128个元素，一共128个元素。
    int repeat = 1;
    // dstLocal: 目的操作数tensor
    // srcLocal: 源操作数tensor
    // srcBlkStride = 1, 在一个repeat中，block间没有空隙。
    // dstRepStride = 1, srcRepStride = 8, repeat间没有空隙。
    AscendC::PairReduceSum<half>(dstLocal, srcLocal, repeat, mask, 1, 1, 8);
    ```

-   PairReduceSum-tensor高维切分计算样例-mask逐bit模式

    ```
    // 设定mask为最多的128个全部元素参与计算
    uint64_t mask[2] = { UINT64_MAX, UINT64_MAX };
    // 每个repeat128个元素，一共128个元素。
    int repeat = 1;
    // dstLocal: 目的操作数tensor
    // srcLocal: 源操作数tensor
    // srcBlkStride = 1, 在一个repeat中，block间没有空隙。
    // dstRepStride = 1, srcRepStride = 8, repeat间没有空隙。
    AscendC::PairReduceSum<half>(dstLocal, srcLocal, repeat, mask, 1, 1, 8);
    ```

-   示例结果

    ```
    输入数据src_gm：
    [1, 1, 1, -1, 2, 2, -1, 2, 
     3, 3, 3, -1, 4, 4, -2, 4,
    ....
    ]
    输出数据dst_gm：
    [2, 0, 4, 1, 6, 2, 8, 2, 
    ....
    ]
    ```

    ```
    输入数据src_gm：
    [-3.441, 7.246, -0.02759, -6.324, 3.693, -7.984, -4.246, 6.332, -3.734, -2.699, -6.91, 7.887, -3.631, 5.219, 6.539, 8.688, 6.523, -6.789, -8.547, 4.258, 1.344, -8.469, -0.9253, -3.914, 3.293, -9.828, 7.082, 5.961, 2.133, 1.959, 3.928, -1.062, 9.18, -1.725, -3.645, 1.457, -2.328, -0.9487, -0.2849, -2.998, -9.281, 3.137, 0.4028, 5.961, -6.25, 2.406, -6.203, -2.699, 4.914, 1.653, -6.383, 6.855, 9.164, 0.6646, -2.854, 3.18, -0.5884, 0.4258, -5.773, -2.152, 4.258, 4.129, -8.719, -8.828, 6.145, 7.387, 1.386, -4.684, 6.324, -1.275, -1.816, 3.357, 6.832, -1.059, -9.852, -8.539, 2.938, -2.002, 9.625, -4.387, -1.309, 8.289, 2.906, -1.035, 7.723, 4.727, -6.477, 2.389, 6.75, -6.688, -0.04248, -6.613, -3.424, 7.145, 4.836, -5.617, -5.855, -5.234, -9.422, -9.852, -8.531, 2.115, 5.109, -8.094, -6.238, 9.898, -6.848, -6.051, 7.109, 4.227, -0.6187, -3.492, -4.352, 1.344, 1.526, 2.572, 2.16, -1.135, 9.812, 1.426, -8, 3.291, -2.039, 5.93, -5.52, -5.156, -9.422, 0.2236]  
    输出数据dst_gm：
    [3.805, -6.352, -4.289, 2.086, -6.434, 0.9766, 1.588, 15.23, -0.2656, -4.289, -7.125, -4.84, -6.535, 13.05, 4.094, 2.865, 7.453, -2.188, -3.277, -3.283, -6.145, 6.363, -3.844, -8.906, 6.566, 0.4727, 9.828, 0.3262, -0.1626, -7.926, 8.391, -17.55, 13.53, -3.297, 5.047, 1.541, 5.773, -18.39, 0.9355, 5.238, 6.98, 1.871, 12.45, -4.086, 0.0625, -6.656, 3.721, -0.7812, -11.09, -19.28, -6.414, -2.984, 3.66, -12.9, 11.34, -4.109, -3.008, 4.098, 1.025, 11.23, -4.711, 3.891, -10.67, -9.195]
    ```

-   完整代码样例

    ```
    #include "kernel_operator.h"
    class KernelReduce {
    public:
        __aicore__ inline KernelReduce() {}
        __aicore__ inline void Init(__gm__ uint8_t* src, __gm__ uint8_t* dstGm)
        {
            srcGlobal.SetGlobalBuffer((__gm__ half*)src);
            dstGlobal.SetGlobalBuffer((__gm__ half*)dstGm);
            pipe.InitBuffer(inQueueSrc, 1, srcDataSize * sizeof(half));
            pipe.InitBuffer(outQueueDst, 1, dstDataSize * sizeof(half));
        }
        __aicore__ inline void Process()
        {
            CopyIn();
            Compute();
            CopyOut();
        }
    private:
        __aicore__ inline void CopyIn()
        {
            AscendC::LocalTensor<half> srcLocal = inQueueSrc.AllocTensor<half>();
            AscendC::DataCopy(srcLocal, srcGlobal, srcDataSize);
            inQueueSrc.EnQue(srcLocal);
        }
        __aicore__ inline void Compute()
        {
            AscendC::LocalTensor<half> srcLocal = inQueueSrc.DeQue<half>();
            AscendC::LocalTensor<half> dstLocal = outQueueDst.AllocTensor<half>();
            half zero(0);
            AscendC::Duplicate(dstLocal, zero, dstDataSize);
            //指令执行部分（替换成上述代码）
            outQueueDst.EnQue<half>(dstLocal);
            inQueueSrc.FreeTensor(srcLocal);
        }
        __aicore__ inline void CopyOut()
        {
            AscendC::LocalTensor<half> dstLocal = outQueueDst.DeQue<half>();
            AscendC::DataCopy(dstGlobal, dstLocal, dstDataSize);
            outQueueDst.FreeTensor(dstLocal);
        }
    private:
        AscendC::TPipe pipe;
        AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueSrc;
        AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueDst;
        AscendC::GlobalTensor<half> srcGlobal, dstGlobal;
        int srcDataSize = 128;
        int dstDataSize = 64;
    };
    extern "C" __global__ __aicore__ void reduce_simple_kernel(__gm__ uint8_t* src, __gm__ uint8_t* dstGm)
    {
        KernelReduce op;
        op.Init(src, dstGm);
        op.Process();
    }
    ```

