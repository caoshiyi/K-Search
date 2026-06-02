# assert<a name="ZH-CN_TOPIC_0000002523303726"></a>

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

## 功能说明<a name="section259105813316"></a>

> **须知：** 
>为了避免部分场景下和标准的assert接口产生冲突，assert\(expr, \_\_gm\_\_ const char \*fmt, Args&&... args\)接口将被废弃，并将在后续版本移除。请不要使用该接口，可使用[ascendc\_assert](ascendc_assert.md)的对应接口作为替代。

基于算子工程开发的算子，可以使用该接口实现CPU/NPU域assert断言功能。算子执行中，如果assert内部条件判断不为真，则输出assert条件并将输入的信息格式化打印在屏幕上。

在算子Kernel侧实现代码中需要增加断言的地方使用assert检查代码，并格式化输出一些调测信息。示例如下：

```
int assertFlag = 10;

assert(assertFlag == 10);
```

> **注意：** 
>assert接口打印功能会对算子实际运行的性能带来一定影响（每一条assert，系统会额外增加一条逻辑判断，具体性能影响取决于代码中assert的使用数量），通常在调测阶段使用。开发者可以按需通过设置ASCENDC\_DUMP=0来关闭打印功能。

NPU域assert打印信息示例如下（DumpHead信息仅在使用自定义算子工程时才会打印）：

```
DumpHead: AIV-0, CoreType=AIV, block dim=8, total_block_num=8, block_remain_len=696, block_initial_space=1024, rsv=0, magic=5aa5bccd
[ASSERT] /home/.../add_custom.cpp:44: Assertion `assertFlag != 10'
```

CPU域assert打印信息示例如下：

```
DumpHead: AIV-0, CoreType=AIV, block dim=8, total_block_num=8, block_remain_len=696, block_initial_space=1024, rsv=0, magic=5aa5bccd
[ASSERT]/home/.../add_custom.cpp:44: Assertion `assertFlag != 10'
```

## 函数原型<a name="section2067518173415"></a>

```
assert(expr)
assert(expr, __gm__ const char *fmt, Args&&... args) // 该接口已废弃，请使用ascendc_assert作为替代
```

## 参数说明<a name="section158061867342"></a>

<a name="zh-cn_topic_0235751031_table33761356"></a>
<table><thead align="left"><tr id="zh-cn_topic_0235751031_row27598891"><th class="cellrowborder" valign="top" width="16.49%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="11.93%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="71.58%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row42461942101815"><td class="cellrowborder" valign="top" width="16.49%" headers="mcps1.1.4.1.1 "><p id="p541413413465"><a name="p541413413465"></a><a name="p541413413465"></a>expr</p>
</td>
<td class="cellrowborder" valign="top" width="11.93%" headers="mcps1.1.4.1.2 "><p id="p1441334144620"><a name="p1441334144620"></a><a name="p1441334144620"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.58%" headers="mcps1.1.4.1.3 "><p id="p84131146466"><a name="p84131146466"></a><a name="p84131146466"></a>assert断言是否终止程序的条件。为true则程序继续执行，为false则终止程序。</p>
</td>
</tr>
<tr id="row163919564263"><td class="cellrowborder" valign="top" width="16.49%" headers="mcps1.1.4.1.1 "><p id="p45208478318"><a name="p45208478318"></a><a name="p45208478318"></a>fmt</p>
</td>
<td class="cellrowborder" valign="top" width="11.93%" headers="mcps1.1.4.1.2 "><p id="p135196472314"><a name="p135196472314"></a><a name="p135196472314"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.58%" headers="mcps1.1.4.1.3 "><p id="p1965816506312"><a name="p1965816506312"></a><a name="p1965816506312"></a>格式控制字符串，包含两种类型的对象：普通字符和转换说明。</p>
<a name="ul419411543310"></a><a name="ul419411543310"></a><ul id="ul419411543310"><li>普通字符将原样不动地打印输出。</li><li>转换说明并不直接输出而是用于控制printf中参数的转换和打印。每个转换说明都由一个百分号字符（%）开始，以转换说明结束，从而说明输出数据的类型 。<div class="p" id="p235913210291"><a name="p235913210291"></a><a name="p235913210291"></a>支持的转换类型包括：<a name="ul541124915329"></a><a name="ul541124915329"></a><ul id="ul541124915329"><li>%d / %i：输出十进制数，支持打印的数据类型：bool/int8_t/int16_t/int32_t/int64_t</li><li>%f：输出实数，支持打印的数据类型：float/half</li><li>%x：输出十六进制整数，支持打印的数据类型：int8_t/int16_t/int32_t/int64_t/uint8_t/uint16_t/uint32_t/uint64_t</li><li>%s：输出字符串</li><li>%u：输出unsigned类型数据，支持打印的数据类型：bool/uint8_t/uint16_t/uint32_t/uint64_t</li><li>%p：输出指针地址</li></ul>
</div>
</li></ul>
<div class="p" id="p733415105257"><a name="p733415105257"></a><a name="p733415105257"></a><strong id="b164621818125011"><a name="b164621818125011"></a><a name="b164621818125011"></a>注意</strong>：<a name="ul63292141989"></a><a name="ul63292141989"></a><ul id="ul63292141989"><li>上文列出的数据类型是NPU域调试支持的数据类型，CPU域调试时，支持的数据类型和C/C++规范保持一致。</li><li>在转换类型为%x，即输出十六进制整数时，NPU域上的输出为64位，CPU域上的输出为32位。</li></ul>
</div>
</td>
</tr>
<tr id="row1818213231324"><td class="cellrowborder" valign="top" width="16.49%" headers="mcps1.1.4.1.1 "><p id="p1563916565265"><a name="p1563916565265"></a><a name="p1563916565265"></a>args</p>
</td>
<td class="cellrowborder" valign="top" width="11.93%" headers="mcps1.1.4.1.2 "><p id="p59396564285"><a name="p59396564285"></a><a name="p59396564285"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.58%" headers="mcps1.1.4.1.3 "><p id="p1257115311337"><a name="p1257115311337"></a><a name="p1257115311337"></a>附加参数，个数和类型可变的参数列表：根据不同的fmt字符串，函数可能需要一系列的附加参数，每个参数包含了一个要被插入的值，替换了fmt参数中指定的每个%标签。参数的个数应与%标签的个数相同。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section43265506459"></a>

-   **该功能仅在如下场景支持：**
    -   通过[Kernel直调](基于样例工程完成Kernel直调.md)方式调用算子。

    -   通过[单算子API调用](单算子API调用.md)方式调用算子。

    -   间接调用单算子API\(aclnnxxx\)接口：Pytorch框架单算子直调的场景。

-   Kernel开发不要包含系统的assert.h，会导致宏定义冲突。
-   该接口调用形式与C语言一致，不需要使用AscendC命名空间。

-   该接口不支持打印除换行符之外的其他转义字符。
-   该接口不支持simulator仿真模式。
-   单次调用本接口打印的数据总量不可超过1MB（还包括少量框架需要的头尾信息，通常可忽略）。使用时应注意，如果超出这个限制，则数据不会被打印。
-   使用自定义算子工程时，存在以下限制：
    -   该接口使用Dump功能，一个算子所有使用Dump功能的接口在每个核上Dump的数据总量不可超过1M。请开发者自行控制待打印的内容数据量，超出则不会打印。
    -   该接口使用空间每个核上不可超过1k。请开发者自行控制待打印的内容数据量，超出则不会打印。

## 调用示例<a name="section82241477610"></a>

```
int assertFlag = 10;
// 断言条件
assert(assertFlag != 10);
```

程序运行时会触发assert，打印效果如下：

```
[ASSERT] /home/.../add_custom.cpp:44: Assertion `assertFlag != 10' 
```

