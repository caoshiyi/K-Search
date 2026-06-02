# MarkStamp<a name="ZH-CN_TOPIC_0000002554423769"></a>

## 产品支持情况<a name="section7433822121115"></a>

<a name="table20433122219118"></a>
<table><thead align="left"><tr id="row543322211114"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p743372211116"><a name="p743372211116"></a><a name="p743372211116"></a><span id="ph043382211112"><a name="ph043382211112"></a><a name="ph043382211112"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p11433122261114"><a name="p11433122261114"></a><a name="p11433122261114"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row24331522101118"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p17301775812"><a name="p17301775812"></a><a name="p17301775812"></a><span id="ph2272194216543"><a name="ph2272194216543"></a><a name="ph2272194216543"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p154339229119"><a name="p154339229119"></a><a name="p154339229119"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section165477552317"></a>

用户通过调用接口，用于在算子执行过程中标记特定位置，便于后期通过流水图分析代码执行路径与性能热点。

## 函数原型<a name="section054795512320"></a>

```
template<pipe_t pipe, uint16_t index>
__aicore__ inline void MarkStamp()
```

## 参数说明<a name="section2054811551732"></a>

<a name="table18824164416323"></a>
<table><thead align="left"><tr id="row4824124418325"><th class="cellrowborder" valign="top" width="50%" id="mcps1.1.3.1.1"><p id="p58240441320"><a name="p58240441320"></a><a name="p58240441320"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="50%" id="mcps1.1.3.1.2"><p id="p2824114412328"><a name="p2824114412328"></a><a name="p2824114412328"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row1882512449324"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.1.3.1.1 "><p id="p88251944153215"><a name="p88251944153215"></a><a name="p88251944153215"></a>pipe</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.1.3.1.2 "><p id="p148291124113616"><a name="p148291124113616"></a><a name="p148291124113616"></a>指定打点所在的pipeline类型。</p>
</td>
</tr>
<tr id="row182624413210"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.1.3.1.1 "><p id="p11826104433211"><a name="p11826104433211"></a><a name="p11826104433211"></a>index</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.1.3.1.2 "><p id="p1826184413328"><a name="p1826184413328"></a><a name="p1826184413328"></a>用户设置的打点的唯一标识id。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section15548145517311"></a>

无

## 约束说明<a name="section654818551936"></a>

-   芯片给index预留了12个bit，取值范围为\[0,4095\]。为方便从打点图中找到对应的代码，建议不要重复使用相同的index。
-   如果在循环中增加了一个MarkStamp指令，每次执行到指令时都会输出一个打点，且index是相同的。
-   如果开发者在两个相邻的VF分别打标记，由于编译器可能会对VF A和VF B做融合，MarkStamp1和MarkStamp2则会被优化掉，不会输出打点。

## 调用示例<a name="section354885515310"></a>

```
mte2_opt();
mte1_opt();
//在算子执行开始处打点
MarkStamp<CUBE, 0>();
//执行核心计算
cube_opt();
//在算子执行结束处打点
MarkStamp<CUBE, 1>();
mte3_opt();
```

