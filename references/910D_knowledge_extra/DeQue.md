# DeQue<a name="ZH-CN_TOPIC_0000002554424349"></a>

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

将Tensor从队列中取出，用于后续处理。

## 函数原型<a name="section620mcpsimp"></a>

-   无需指定源和目的位置
    -   non-inplace接口：将入队的LocalTensor地址从队列中取出赋值给新创建的Tensor并返回

        ```
        template <typename T>
        __aicore__ inline LocalTensor<T> DeQue()
        ```

    -   inplace接口：通过出参的方式返回，可以减少Tensor反复创建的开销，具体使用指导可参考[如何使用Tensor原地操作提升算子性能](如何使用Tensor原地操作提升算子性能.md)。

        ```
        template <typename T>
        __aicore__ inline void DeQue(LocalTensor<T>& tensor)
        ```

-   需要指定源和目的位置

    通过[TQueBind](TQueBind.md)绑定VECIN和VECOUT可实现VECIN和VECOUT内存复用，如下接口用于存在Vector计算的场景下实现复用，在出队时需要指定源和目的位置；不存在Vector计算的场景下可直接调用LocalTensor<T\> DeQue\(\)出队接口。

    ```
    template <TPosition srcUserPos, TPosition dstUserPos, typename T>
    __aicore__ inline LocalTensor<T> DeQue()
    ```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table1550165916920"></a>
<table><thead align="left"><tr id="row115015591391"><th class="cellrowborder" valign="top" width="12.139999999999999%" id="mcps1.2.3.1.1"><p id="p12501159099"><a name="p12501159099"></a><a name="p12501159099"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="87.86%" id="mcps1.2.3.1.2"><p id="p85019592918"><a name="p85019592918"></a><a name="p85019592918"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row1550117591914"><td class="cellrowborder" valign="top" width="12.139999999999999%" headers="mcps1.2.3.1.1 "><p id="p185019592913"><a name="p185019592913"></a><a name="p185019592913"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="87.86%" headers="mcps1.2.3.1.2 "><p id="p12101541625"><a name="p12101541625"></a><a name="p12101541625"></a><span>Tensor的数据类型。</span></p>
</td>
</tr>
<tr id="row1629562101416"><td class="cellrowborder" valign="top" width="12.139999999999999%" headers="mcps1.2.3.1.1 "><p id="p174916484213"><a name="p174916484213"></a><a name="p174916484213"></a>srcUserPos</p>
</td>
<td class="cellrowborder" valign="top" width="87.86%" headers="mcps1.2.3.1.2 "><p id="p24912414422"><a name="p24912414422"></a><a name="p24912414422"></a>用户指定队列的src position，当前只支持如下通路：GM-&gt;VECIN/VECOUT-&gt;GM。</p>
</td>
</tr>
<tr id="row1780112318146"><td class="cellrowborder" valign="top" width="12.139999999999999%" headers="mcps1.2.3.1.1 "><p id="p1469019111420"><a name="p1469019111420"></a><a name="p1469019111420"></a>dstUserPos</p>
</td>
<td class="cellrowborder" valign="top" width="87.86%" headers="mcps1.2.3.1.2 "><p id="p96907110428"><a name="p96907110428"></a><a name="p96907110428"></a>用户指定队列的dst position，当前只支持如下通路：GM-&gt;VECIN/VECOUT-&gt;GM。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  参数说明

<a name="table181221135162517"></a>
<table><thead align="left"><tr id="row151221135112520"><th class="cellrowborder" valign="top" width="12.471247124712471%" id="mcps1.2.4.1.1"><p id="p1353754532512"><a name="p1353754532512"></a><a name="p1353754532512"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="12.591259125912593%" id="mcps1.2.4.1.2"><p id="p1253774516259"><a name="p1253774516259"></a><a name="p1253774516259"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="74.93749374937492%" id="mcps1.2.4.1.3"><p id="p1653710452259"><a name="p1653710452259"></a><a name="p1653710452259"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row12122235102511"><td class="cellrowborder" valign="top" width="12.471247124712471%" headers="mcps1.2.4.1.1 "><p id="p1537164502512"><a name="p1537164502512"></a><a name="p1537164502512"></a>tensor</p>
</td>
<td class="cellrowborder" valign="top" width="12.591259125912593%" headers="mcps1.2.4.1.2 "><p id="p1653714592515"><a name="p1653714592515"></a><a name="p1653714592515"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="74.93749374937492%" headers="mcps1.2.4.1.3 "><p id="p159171759152612"><a name="p159171759152612"></a><a name="p159171759152612"></a>inplace接口需要通过出参的方式返回Tensor。</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section633mcpsimp"></a>

-   对空队列执行DeQue是一种异常行为，会在CPU调测时报错。
-   non-inplace接口和指定源和目的位置的接口，需要将TQueBind的depth模板参数设置为非零值；inplace接口，需要将TQueBind的depth模板参数设置为0。

## 返回值说明<a name="section640mcpsimp"></a>

non-inplace接口和指定源和目的位置的接口返回值为从队列中取出的[LocalTensor](LocalTensor.md)；inplace接口没有返回值。

## 调用示例<a name="section642mcpsimp"></a>

```
// 接口: DeQue Tensor
AscendC::TPipe pipe;
AscendC::TQue<AscendC::TPosition::VECOUT, 4> que;
int num = 4;
int len = 1024;
pipe.InitBuffer(que, num, len);
AscendC::LocalTensor<half> tensor1 = que.AllocTensor<half>();
que.EnQue(tensor1);
AscendC::LocalTensor<half> tensor2 = que.DeQue<half>(); // 将tensor从VECOUT的Queue中搬出
// 接口: DeQue Tensor，指定特定的Src/Dst position
AscendC::TPipe pipe;
AscendC::TQueBind<AscendC::TPosition::VECIN, AscendC::TPosition::VECOUT, 1> que;
int num = 4;
int len = 1024;
pipe.InitBuffer(que, num, len);
AscendC::LocalTensor<half> tensor1 = que.AllocTensor<half>();
que.EnQue<AscendC::TPosition::GM, AscendC::TPosition::VECIN, half>(tensor1);
// 将tensor从VECIN的Queue中搬出
AscendC::LocalTensor<half> tensor2 = que.DeQue<AscendC::TPosition::GM, AscendC::TPosition::VECIN, half>(); 
// inplace接口
AscendC::TPipe pipe;
AscendC::TQue<AscendC::TPosition::VECOUT, 0> que;
int num = 2;
int len = 1024;
pipe.InitBuffer(que, num, len);
AscendC::LocalTensor<half> tensor1;
que.AllocTensor<half>(tensor1);
que.EnQue(tensor1);
que.DeQue<half>(tensor1); // 将tensor从VECOUT的Queue中搬出
que.FreeTensor<half>(tensor1);
```

