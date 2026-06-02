# PhiloxRandom<a name="ZH-CN_TOPIC_0000002523304792"></a>

## 产品支持情况<a name="section1550532418810"></a>

<a name="table38301303189"></a>
<table><thead align="left"><tr id="row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p1883113061818"><a name="p1883113061818"></a><a name="p1883113061818"></a><span id="ph20833205312295"><a name="ph20833205312295"></a><a name="ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p783113012187"><a name="p783113012187"></a><a name="p783113012187"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row1272474920205"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p12300735171314"><a name="p12300735171314"></a><a name="p12300735171314"></a><span id="ph730011352138"><a name="ph730011352138"></a><a name="ph730011352138"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p37256491200"><a name="p37256491200"></a><a name="p37256491200"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section618mcpsimp"></a>

基于Philox随机数生成算法，给定随机数种子，生成若干的随机数。

Philox随机数生成的核心算法是一个基于记数的伪随机数生成算法，输入为一个128bit的记数器C，两个32bit的key（k0和k1），输出为4个32bit的整数。

## 函数原型<a name="section620mcpsimp"></a>

-   连续模式

    ```
    template <uint16_t Rounds = 7, typename T>
    __aicore__ inline void PhiloxRandom(const LocalTensor<T>& dstLocal, const PhiloxKey& philoxKey, const PhiloxCounter& philoxCounter, uint16_t count)
    ```

-   stride模式

    ```
    template <uint16_t Rounds = 7, typename T>
    __aicore__ inline void PhiloxRandom(const LocalTensor<T>& dstLocal, const PhiloxKey& philoxKey, const PhiloxCounter& philoxCounter, const PhiloxRandomParams& params)
    ```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table4835205712588"></a>
<table><thead align="left"><tr id="row118356578583"><th class="cellrowborder" valign="top" width="18.19%" id="mcps1.2.3.1.1"><p id="p48354572582"><a name="p48354572582"></a><a name="p48354572582"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="81.81%" id="mcps1.2.3.1.2"><p id="p583535795817"><a name="p583535795817"></a><a name="p583535795817"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row1835857145817"><td class="cellrowborder" valign="top" width="18.19%" headers="mcps1.2.3.1.1 "><p id="p5835457165816"><a name="p5835457165816"></a><a name="p5835457165816"></a>Rounds</p>
</td>
<td class="cellrowborder" valign="top" width="81.81%" headers="mcps1.2.3.1.2 "><p id="p168351657155818"><a name="p168351657155818"></a><a name="p168351657155818"></a>Philox算法内部实现迭代次数，支持取值7或10。</p>
</td>
</tr>
<tr id="row18835145716587"><td class="cellrowborder" valign="top" width="18.19%" headers="mcps1.2.3.1.1 "><p id="p1383515717581"><a name="p1383515717581"></a><a name="p1383515717581"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="81.81%" headers="mcps1.2.3.1.2 "><p id="p1082824863813"><a name="p1082824863813"></a><a name="p1082824863813"></a>目的操作数数据类型，支持的数据类型为：uint32_t、int32_t、float。</p>
<p id="p6323338183913"><a name="p6323338183913"></a><a name="p6323338183913"></a>其中uint32_t/int32_t为数据类型范围内的均匀分布，float为0-1范围内的均匀分布。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  参数说明

<a name="zh-cn_topic_0235751031_table33761356"></a>
<table><thead align="left"><tr id="zh-cn_topic_0235751031_row27598891"><th class="cellrowborder" valign="top" width="18.54%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="10.05%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="71.41%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row42461942101815"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p284425844311"><a name="p284425844311"></a><a name="p284425844311"></a>dstLocal</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p158449584436"><a name="p158449584436"></a><a name="p158449584436"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p6389114616514"><a name="p6389114616514"></a><a name="p6389114616514"></a>目的操作数。</p>
<p id="p17644848194211"><a name="p17644848194211"></a><a name="p17644848194211"></a><span id="zh-cn_topic_0000002523303824_ph173308471594"><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><span id="zh-cn_topic_0000002523303824_ph9902231466"><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816"><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p1176131104213"><a name="p1176131104213"></a><a name="p1176131104213"></a><span id="ph171761731114217"><a name="ph171761731114217"></a><a name="ph171761731114217"></a>LocalTensor的起始地址需要32字节对齐。</span></p>
</td>
</tr>
<tr id="row2137145181815"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p6844125874315"><a name="p6844125874315"></a><a name="p6844125874315"></a>philoxKey</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p128442058144312"><a name="p128442058144312"></a><a name="p128442058144312"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p0534135117111"><a name="p0534135117111"></a><a name="p0534135117111"></a>随机数种子。两个32bit的key，定义如下：</p>
<a name="screen1284915182119"></a><a name="screen1284915182119"></a><pre class="screen" codetype="Cpp" id="screen1284915182119">using PhiloxKey = uint32_t[2];</pre>
</td>
</tr>
<tr id="row891912431168"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p195756503168"><a name="p195756503168"></a><a name="p195756503168"></a>philoxCounter</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p457515071618"><a name="p457515071618"></a><a name="p457515071618"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p146281948144619"><a name="p146281948144619"></a><a name="p146281948144619"></a>随机数种子。一个128bit的记数器C（由4个32bit组成），定义如下：</p>
<a name="screen1262844844610"></a><a name="screen1262844844610"></a><pre class="screen" codetype="Cpp" id="screen1262844844610">using PhiloxCounter = uint32_t[4];</pre>
</td>
</tr>
<tr id="row148313330454"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p16494238134512"><a name="p16494238134512"></a><a name="p16494238134512"></a>count</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p14949383456"><a name="p14949383456"></a><a name="p14949383456"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p12831233194516"><a name="p12831233194516"></a><a name="p12831233194516"></a>生成目的操作数的元素个数。</p>
</td>
</tr>
<tr id="row250163914199"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p98451586430"><a name="p98451586430"></a><a name="p98451586430"></a>params</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p20845205894317"><a name="p20845205894317"></a><a name="p20845205894317"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p1355205134617"><a name="p1355205134617"></a><a name="p1355205134617"></a>stride模式计算所需的参数信息。PhiloxRandomParams类型，定义如下：</p>
<a name="screen155585184616"></a><a name="screen155585184616"></a><pre class="screen" codetype="Cpp" id="screen155585184616">struct PhiloxRandomParams {
   uint32_t stride;  // 两行元素之间的间隔
   uint32_t row;     // 表示生成的行数
   uint32_t column;  // 表示生成的每一行的元素个数
}</pre>
<a name="ul1761mcpsimp"></a><a name="ul1761mcpsimp"></a><ul id="ul1761mcpsimp"><li>row * column大于0，不大于LocalTensor的大小。</li><li>column % 4 == 0，stride % 4 == 0，stride &gt;= column。</li></ul>
</td>
</tr>
</tbody>
</table>

**图 1**  PhiloxRandom示意图<a name="fig1045614325812"></a>  
<!-- img2text -->
```
                           column
                <────────────────────────>
┌───────────────────────────────────────────────────────────────┐
│                                                               │
│  stride                                                       │
│  <──────>                                                     │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │   0      1      2      3      4      5      6      7   │  │
│  │                                                         │  │
│  │   8      9     10     11     12     13     14     15   │  │
│  │                                                         │  │
│  │  16     17     18     16     17     16     17     16   │  │
│  │                                                         │  │
│  │  24     25     26     27     28     29     30     31   │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                               │
│  row                                                          │
│  ↑                                                            │
│  │                                                            │
│  │                                                            │
│  ↓                                                            │
└───────────────────────────────────────────────────────────────┘
```

上图是一个生成随机数的示意图。

-   连续模式下使用philoxCounter=\{0, 0, 0, 0\}，count=32来生成32个随机数。
-   stride模式下可按列分两次生成，调用两次接口。第一次调用参数为philoxCounter=\{0, 0, 0, 0\}，stride=8，row=4，column=4；第二次调用参数为philoxCounter=\{1, 0, 0, 0\}（每次记数器C自增会生成128bit的随机数），stride=8，row=4，column=4。

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

无

## 调用示例<a name="section642mcpsimp"></a>

-   接口使用样例

    ```
    // philoxKey={0,0}, philoxCounter={0,0,0,0}, params={1024, 8, 1024}
    LocalTensor<uint32_t> dstLocal = outQueue.AllocTensor<uint32_t>();  
    PhiloxRandom<7>(dstLocal, {0, 0}, {0, 0, 0, 0},{1024, 8, 1024});
    ```

-   完整样例

    ```
    #include "kernel_operator.h"
    
    template <uint16_t Rounds, typename srcType>
    class KernelPhiloxStride {
    public:
        __aicore__ inline KernelPhiloxStride() {}
        __aicore__ inline void Init(GM_ADDR dstGm, uint32_t paramStride, uint32_t paramRow, uint32_t paramColumn)
        {
            stride = paramStride;
            row = paramRow;
            column = paramColumn;
            count = row * column;
            const int alginSize = AscendC::GetDataBlockSizeInBytes() / sizeof(srcType);
            dstSize = (count + alginSize - 1) / alginSize * alginSize;
            dstGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ srcType *>(dstGm), dstSize);
            pipe.InitBuffer(outQueue, 1, dstSize * sizeof(srcType));
        }
        __aicore__ inline void Process(uint32_t seed0, uint32_t seed1, uint32_t seed2, uint32_t seed3, uint32_t seed4,
            uint32_t seed5)
        {
            Compute(seed0, seed1, seed2, seed3, seed4, seed5);
            CopyOut();
        }
    private:
        __aicore__ inline void Compute(uint32_t seed0, uint32_t seed1, uint32_t seed2, uint32_t seed3, uint32_t seed4,
            uint32_t seed5)
        {
            AscendC::LocalTensor<srcType> dstLocal = outQueue.AllocTensor<srcType>();
            AscendC::PhiloxRandom<Rounds>(dstLocal, { seed0, seed1 }, { seed2, seed3, seed4, seed5 },
                { stride, row, column });
            outQueue.EnQue<srcType>(dstLocal);
        }
        __aicore__ inline void CopyOut()
        {
            AscendC::LocalTensor<srcType> dstLocal = outQueue.DeQue<srcType>();
            AscendC::DataCopy(dstGlobal, dstLocal, dstSize);
            outQueue.FreeTensor(dstLocal);
        }
    private:
        AscendC::GlobalTensor<srcType> dstGlobal;
        AscendC::TPipe pipe;
        AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueue;
        uint32_t count;
        uint32_t stride;
        uint32_t row;
        uint32_t column;
        uint32_t dstSize;
    };
    
    extern "C" __global__ __aicore__ void philox_kernel_stride(GM_ADDR dstGm)
    {
        KernelPhiloxStride<7, uint32_t> op;
        op.Init(dstGm, 1024, 8, 1024);
        op.Process(0, 0, 0, 0, 0, 0);
    }
    ```

