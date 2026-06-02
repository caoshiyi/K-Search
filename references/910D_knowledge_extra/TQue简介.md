# TQue简介<a name="ZH-CN_TOPIC_0000002554423859"></a>

流水任务之间通过队列（Queue）完成任务间通信和同步。TQue是用来执行队列相关操作、管理相关资源的数据结构。TQue继承自TQueBind父类，继承关系如下：

<!-- img2text -->
```text
           父类
┌──────────────┐
│   TQueBind   │
└──────────────┘
        ▲
        │
┌──────────────┐
│     TQue     │
└──────────────┘
           子类
```

## 模板参数<a name="section18341144185913"></a>

```
template <TPosition pos, int32_t depth, auto mask = 0> class TQue{...};
```

**表 1**  TQue模板参数介绍

<a name="table1550165916920"></a>
<table><thead align="left"><tr id="row115015591391"><th class="cellrowborder" valign="top" width="14.099999999999998%" id="mcps1.2.3.1.1"><p id="p12501159099"><a name="p12501159099"></a><a name="p12501159099"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="85.9%" id="mcps1.2.3.1.2"><p id="p85019592918"><a name="p85019592918"></a><a name="p85019592918"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row1550117591914"><td class="cellrowborder" valign="top" width="14.099999999999998%" headers="mcps1.2.3.1.1 "><p id="p185019592913"><a name="p185019592913"></a><a name="p185019592913"></a>pos</p>
</td>
<td class="cellrowborder" valign="top" width="85.9%" headers="mcps1.2.3.1.2 "><p id="p35011591693"><a name="p35011591693"></a><a name="p35011591693"></a>队列逻辑位置，可以为<span>VECIN、VECOUT、</span>A1<span>、</span>A2<span>、</span>B1<span>、</span>B2<span>、</span>CO1<span>、</span>CO2。<span>关于TPosition的具体介绍请参考</span><a href="TPosition.md">TPosition</a>。</p>
</td>
</tr>
<tr id="row12501859799"><td class="cellrowborder" valign="top" width="14.099999999999998%" headers="mcps1.2.3.1.1 "><p id="p1650113599915"><a name="p1650113599915"></a><a name="p1650113599915"></a>depth</p>
</td>
<td class="cellrowborder" valign="top" width="85.9%" headers="mcps1.2.3.1.2 "><p id="p8510214185418"><a name="p8510214185418"></a><a name="p8510214185418"></a>队列的深度表示该队列可以连续进行入队/出队的次数，在代码运行时，对同一个队列有n次连续的EnQue（中间没有DeQue），那么该队列的深度就需要设置为n。</p>
<p id="p7674135716528"><a name="p7674135716528"></a><a name="p7674135716528"></a>注意，这里的队列深度和double buffer无关，队列机制用于实现流水线并行，double buffer在此基础上进一步提高流水线的利用率。即使队列的深度为1，仍可以开启double buffer。</p>
<p id="p1334591855410"><a name="p1334591855410"></a><a name="p1334591855410"></a>非Tensor原地操作的场景下，队列的深度设置为1时，编译器对这种场景做了特殊优化，性能通常更好，<strong id="b0597122135416"><a name="b0597122135416"></a><a name="b0597122135416"></a>推荐设置为1</strong>。</p>
<p id="p1167810565476"><a name="p1167810565476"></a><a name="p1167810565476"></a><a href="如何使用Tensor原地操作提升算子性能.md">Tensor原地操作</a>的场景下，需要设置为0。</p>
<a name="ul26757572527"></a><a name="ul26757572527"></a><ul id="ul26757572527"><li>如下样例中队列没有连续入队，队列的深度设置为1。<a name="screen17724335125812"></a><a name="screen17724335125812"></a><pre class="screen" codetype="Cpp" id="screen17724335125812">a1 = que.AllocTensor(); 
que.EnQue(a1);
a1 = que.DeQue();
que.FreeTensor(a1);</pre>
</li></ul>
<a name="ul12675135785214"></a><a name="ul12675135785214"></a><ul id="ul12675135785214"><li>如下样例中队列连续2次入队，队列的深度应设置为2，仅在极少数preload场景（比如连续搬入两份数据，计算处理一份，完成后再搬入一份，然后计算处理提前搬入的一份...）可能会使用。其他情况下均不推荐depth &gt;= 2 。<a name="screen11973229582"></a><a name="screen11973229582"></a><pre class="screen" codetype="Cpp" id="screen11973229582">a1 = que.AllocTensor(); 
a2 = que.AllocTensor();
que.EnQue(a1);
que.EnQue(a2);
a1 = que.DeQue();
a2 = que.DeQue(); 
que.FreeTensor(a1);
que.FreeTensor(a2);</pre>
</li></ul>
</td>
</tr>
<tr id="row3501135910920"><td class="cellrowborder" valign="top" width="14.099999999999998%" headers="mcps1.2.3.1.1 "><p id="p6501175912914"><a name="p6501175912914"></a><a name="p6501175912914"></a>mask</p>
</td>
<td class="cellrowborder" valign="top" width="85.9%" headers="mcps1.2.3.1.2 "><a name="ul4317543497"></a><a name="ul4317543497"></a><ul id="ul4317543497"><li>mask是const TQueConfig*类型时，TQueConfig结构定义和参数说明如下:<a name="screen13896155731115"></a><a name="screen13896155731115"></a><pre class="screen" codetype="Cpp" id="screen13896155731115">struct TQueConfig {
    bool scmBlockGroup = false;  // TSCM相关参数，预留参数，默认为false
    uint32_t bufferLen = 0;  // 与InitBuffer时输入的len参数保持一致，可以在编译期做性能优化，<strong id="b1084715241371"><a name="b1084715241371"></a><a name="b1084715241371"></a>传0表示在InitBuffer时做资源分配</strong>。
    uint32_t bufferNumber = 0;  // 与InitBuffer时输入的num参数保持一致，可以在编译期做性能优化，<strong id="b132591319977"><a name="b132591319977"></a><a name="b132591319977"></a>传0表示在InitBuffer时做资源分配</strong>。
    uint32_t consumerSize = 0;  // 预留参数
    TPosition consumer[8] = {}; // 预留参数
    bool enableStaticEvtId = false; // 预留参数
    bool enableLoopQueue = false;   // 预留参数
};</pre>
</li></ul>
</td>
</tr>
</tbody>
</table>

## TQue Buffer限制<a name="section466543213575"></a>

由于TQue分配的Buffer存储着同步事件eventID，故同一个TPosition上TQue Buffer的数量与硬件的同步事件eventID有关。

QUE的Buffer数量最大也分别为8个或4个，即能插入的同步事件的个数为8个或4个。当用TPipe的InitBuffer申请TQue时，会受到Buffer数量的限制，TQue能申请到的最大个数分别为8个或4个。

如果同时使用的QUE Buffer超出限制，则无法再申请TQue。如果想要继续申请，可以调用FreeAllEvent接口来释放一些暂时不用的TQue。在使用完对应TQue后，用该接口释放对应队列中的所有事件，之后便可再次申请TQue。样例如下：

```
// 能申请的VECIN position上的buffer数量最大为8。如果超出该限制，在后续使用AllocTensor/FreeTensor可能会出现分配资源失败。故当不开启double buffer时，此时最多能申请8个TQue。
AscendC::TPipe pipe;
int len = 1024;
AscendC::TQue<AscendC::TPosition::VECIN, 1> que0;
AscendC::TQue<AscendC::TPosition::VECIN, 1> que1;
AscendC::TQue<AscendC::TPosition::VECIN, 1> que2;
AscendC::TQue<AscendC::TPosition::VECIN, 1> que3;
AscendC::TQue<AscendC::TPosition::VECIN, 1> que4;
AscendC::TQue<AscendC::TPosition::VECIN, 1> que5;
AscendC::TQue<AscendC::TPosition::VECIN, 1> que6;
AscendC::TQue<AscendC::TPosition::VECIN, 1> que7;
 
pipe.InitBuffer(que0, 1, len);
pipe.InitBuffer(que1, 1, len);
pipe.InitBuffer(que2, 1, len);
pipe.InitBuffer(que3, 1, len);
pipe.InitBuffer(que4, 1, len);
pipe.InitBuffer(que5, 1, len);
pipe.InitBuffer(que6, 1, len);
pipe.InitBuffer(que7, 1, len);
 
// 如果开启double buffer，此时每一个TQue分配的内存块个数为2，故最多只能申请4个TQue。
TPipe pipe;
int len = 1024;
AscendC::TQue<AscendC::TPosition::VECIN, 1> que0;
AscendC::TQue<AscendC::TPosition::VECIN, 1> que1;
AscendC::TQue<AscendC::TPosition::VECIN, 1> que2;
AscendC::TQue<AscendC::TPosition::VECIN, 1> que3;
 
pipe.InitBuffer(que0, 2, len);
pipe.InitBuffer(que1, 2, len);
pipe.InitBuffer(que2, 2, len);
pipe.InitBuffer(que3, 2, len);
 
// 如果TQue个数已达最大值，可以调用FreeAllEvent接口来继续申请TQue。
AscendC::TPipe pipe;
int len = 1024;
AscendC::TQue<AscendC::TPosition::VECIN, 1> que0;
pipe.InitBuffer(que0, 1, len);
AscendC::LocalTensor<half> tensor1 = que0.AllocTensor<half>();
que0.EnQue(tensor1);
tensor1 = que0.DeQue<half>(); // 将tensor从VECOUT的Queue中搬出
que0.FreeTensor<half>(tensor1);
que0.FreeAllEvent(); // 释放que0的所有同步事件，之后可继续申请TQue
AscendC::TQue<AscendC::TPosition::VECIN, 1> que1;
pipe.InitBuffer(que1, 1, len);
```

