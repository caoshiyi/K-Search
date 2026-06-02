# Iterate<a name="ZH-CN_TOPIC_0000002523304852"></a>

## 产品支持情况<a name="section1550532418810"></a>

<a name="table1334714391211"></a>
<table><thead align="left"><tr id="row1334743121213"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p834713321216"><a name="p834713321216"></a><a name="p834713321216"></a><span id="ph834783101215"><a name="ph834783101215"></a><a name="ph834783101215"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p2347234127"><a name="p2347234127"></a><a name="p2347234127"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row113472312122"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p234710320128"><a name="p234710320128"></a><a name="p234710320128"></a><span id="ph103471336127"><a name="ph103471336127"></a><a name="ph103471336127"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p4751940181211"><a name="p4751940181211"></a><a name="p4751940181211"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section618mcpsimp"></a>

每调用一次Iterate，会计算出一块baseM \* baseN的C矩阵。接口内部会维护迭代进度，调用一次后会对A、B矩阵首地址进行偏移。默认以先M轴再N轴的迭代顺序，也可以通过调整tiling参数iterateOrder，转换为先N轴再M轴的迭代顺序。当传入数据未对齐，存在尾块时，会在最后一次迭代输出尾块的计算结果。

一次Iterate矩阵乘的结果C矩阵存放在[逻辑位置CO1](基础知识.md#zh-cn_topic_0000001622194138_zh-cn_topic_0000001455771256_li42261523152714)的内存中，对于CO1内存中计算结果的获取，当前支持如下两种方式：

-   用户无需自行管理存放矩阵乘结果的CO1内存的申请和释放，由Matmul API内部实现管理。调用[接口内部管理CO1](#li135771283591)的Iterate函数原型后，调用[GetTensorC](GetTensorC.md)接口完成CO1上计算结果的搬出。
-   用户可以灵活自主地控制矩阵乘计算结果的搬运，例如将多次Iterate计算的矩阵乘结果缓存在CO1内存中，在需要搬出该结果时，一次性搬出多块baseM \* baseN的C矩阵。这种灵活搬运场景下，用户需要提前申请CO1的内存，调用[用户自主管理CO1](#li4843165185812)的Iterate函数原型后，一次Iterate的计算结果会输出到用户申请的CO1内存上。在需要搬出计算结果时，调用[Fixpipe](Fixpipe.md)接口搬运CO1上的结果，完成后释放申请的CO1内存。具体示例请参考[用户自主管理CO1的矩阵乘场景](Matmul使用说明.md#li12765161915318)。

## 函数原型<a name="section620mcpsimp"></a>

-   <a name="li135771283591"></a>接口内部管理CO1

    ```
    template <bool sync = true> __aicore__ inline bool Iterate(bool enPartialSum = false)
    ```

-   <a name="li4843165185812"></a>用户自主管理CO1

    ```
    template <bool sync = true, typename T> __aicore__ inline bool Iterate(bool enPartialSum, const LocalTensor<T>& localCmatrix)
    ```

## 参数说明<a name="section1656111295569"></a>

**表 1**  模板参数说明

<a name="table1656122919564"></a>
<table><thead align="left"><tr id="row17561172920562"><th class="cellrowborder" valign="top" width="17.04%" id="mcps1.2.3.1.1"><p id="p75611299565"><a name="p75611299565"></a><a name="p75611299565"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="82.96%" id="mcps1.2.3.1.2"><p id="p195611129105612"><a name="p195611129105612"></a><a name="p195611129105612"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row556115297566"><td class="cellrowborder" valign="top" width="17.04%" headers="mcps1.2.3.1.1 "><p id="p125616298567"><a name="p125616298567"></a><a name="p125616298567"></a>sync</p>
</td>
<td class="cellrowborder" valign="top" width="82.96%" headers="mcps1.2.3.1.2 "><p id="p756111299566"><a name="p756111299566"></a><a name="p756111299566"></a>迭代获取C矩阵分片的过程分为同步和异步两种模式。通过该参数设置同步或者异步模式：同步模式设置为true；异步模式设置为false。默认为同步模式。具体模式的介绍和使用方法请参考<a href="GetTensorC.md">GetTensorC</a>。</p>
</td>
</tr>
<tr id="row19797956172116"><td class="cellrowborder" valign="top" width="17.04%" headers="mcps1.2.3.1.1 "><p id="p1885105792120"><a name="p1885105792120"></a><a name="p1885105792120"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="82.96%" headers="mcps1.2.3.1.2 "><p id="p1388505714215"><a name="p1388505714215"></a><a name="p1388505714215"></a>用户申请的CO1内存上LocalTensor的数据类型，即矩阵乘输出的C矩阵的数据类型。当前支持的数据类型为float、int32_t。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  接口内部管理CO1的函数参数说明

<a name="table95611529195618"></a>
<table><thead align="left"><tr id="row3562132916562"><th class="cellrowborder" valign="top" width="14.99%" id="mcps1.2.4.1.1"><p id="p956262911563"><a name="p956262911563"></a><a name="p956262911563"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="12.02%" id="mcps1.2.4.1.2"><p id="p256292912568"><a name="p256292912568"></a><a name="p256292912568"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="72.99%" id="mcps1.2.4.1.3"><p id="p17562129145619"><a name="p17562129145619"></a><a name="p17562129145619"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row125621729205612"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p956242916567"><a name="p956242916567"></a><a name="p956242916567"></a>enPartialSum</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p35624293566"><a name="p35624293566"></a><a name="p35624293566"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p11562102995617"><a name="p11562102995617"></a><a name="p11562102995617"></a>是否将矩阵乘的结果累加于现有的CO1数据，默认值为false。在L0C累加时，只支持C矩阵规格为singleCoreM==baseM &amp;&amp; singleCoreN==baseN。</p>
</td>
</tr>
</tbody>
</table>

**表 3**  用户自主管理CO1的函数参数说明

<a name="table9646134355611"></a>
<table><thead align="left"><tr id="row964714433565"><th class="cellrowborder" valign="top" width="15.190000000000001%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="11.82%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="72.99%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row16491543185617"><td class="cellrowborder" valign="top" width="15.190000000000001%" headers="mcps1.2.4.1.1 "><p id="p3225452133"><a name="p3225452133"></a><a name="p3225452133"></a>enPartialSum</p>
</td>
<td class="cellrowborder" valign="top" width="11.82%" headers="mcps1.2.4.1.2 "><p id="p52251252339"><a name="p52251252339"></a><a name="p52251252339"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p01481439165610"><a name="p01481439165610"></a><a name="p01481439165610"></a>是否将矩阵乘的结果累加于现有的CO1数据。在L0C累加时，只支持C矩阵规格为singleCoreM==baseM &amp;&amp; singleCoreN==baseN。</p>
</td>
</tr>
<tr id="row8603913417"><td class="cellrowborder" valign="top" width="15.190000000000001%" headers="mcps1.2.4.1.1 "><p id="p960796418"><a name="p960796418"></a><a name="p960796418"></a>localCmatrix</p>
</td>
<td class="cellrowborder" valign="top" width="11.82%" headers="mcps1.2.4.1.2 "><p id="p176016924114"><a name="p176016924114"></a><a name="p176016924114"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p17600954113"><a name="p17600954113"></a><a name="p17600954113"></a>由用户申请的CO1上的LocalTensor内存，用于存放矩阵乘的计算结果。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

false：单核上的所有数据全部算完。

true：数据仍在迭代计算中。

## 约束说明<a name="section633mcpsimp"></a>

-   当使能MixDualMaster（双主模式）场景时，即模板参数[enableMixDualMaster](MatmulConfig.md#p9218181073719)设置为true，不支持使用该接口。
-   对于用户自主管理CO1的Iterate函数，创建Matmul对象时，必须定义C矩阵的内存逻辑位置为TPosition::CO1、数据排布格式为CubeFormat::NZ、数据类型为float或int32\_t。

## 调用示例<a name="section1665082013318"></a>

同步模式及异步模式的简单调用示例如下，更多完整算子样例请参考[IterateAll异步样例](https://gitcode.com/cann/asc-devkit/tree/master/examples/03_libraries/01_matrix/matmul_async_iterate_all)、[Iterate异步场景样例](https://gitcode.com/cann/asc-devkit/tree/master/examples/03_libraries/01_matrix/matmul_async_iterate)、[自主管理CO1的算子样例](https://gitcode.com/cann/asc-devkit/tree/master/examples/03_libraries/01_matrix/matmul_l0c_extend)。

```
// 同步模式样例
while (mm.Iterate()) {   
    mm.GetTensorC(ubCmatrix); 
}

// 异步模式样例
mm.template Iterate<false>();
// …… ……其它计算
for (int i = 0; i < singleM/baseM*singleN/baseN; ++i) {   
    mm.template GetTensorC<false>(ubCmatrix); 
}
```

