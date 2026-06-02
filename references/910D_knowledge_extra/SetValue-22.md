# SetValue<a name="ZH-CN_TOPIC_0000002523344846"></a>

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

设置GlobalTensor相应偏移位置的值。

-   因为硬件实现不同，其与通用CPU标量赋值操作存在差异。SetValue赋值操作首先改写的是每个AI Core内部的DCache， 不会立刻写出到Global Memory，且后续写出时以Cache Line（64B）为单位。使用该接口之前必须了解DCache结构和Cache一致性原理（参见[DataCacheCleanAndInvalid](DataCacheCleanAndInvalid.md)），否则可能存在误用的情况，**请谨慎使用。**
-   调用SetValue后，首先改写的是每个AI Core内部的DCache，如果需要立即写出到Global Memory，需要在调用此接口后，再调用[DataCacheCleanAndInvalid](DataCacheCleanAndInvalid.md)，确保DCache与Global Memory的Cache一致性。
-   多核操作GM地址时，要求不同核操作的地址（通过offset参数设置元素偏移，可以转换为地址）至少有Cache Line大小的偏移，否则会出现多核数据随机覆盖。同时需要考虑地址对齐（64B）的问题。详细内容请参考[调用示例](DataCacheCleanAndInvalid.md#section837496171220)。

## 函数原型<a name="section620mcpsimp"></a>

```
__aicore__ inline void SetValue(const uint64_t offset, PrimType value)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  参数说明

<a name="zh-cn_topic_0235751031_table33761356"></a>
<table><thead align="left"><tr id="zh-cn_topic_0235751031_row27598891"><th class="cellrowborder" valign="top" width="13.94%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="12.98%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="73.08%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row42461942101815"><td class="cellrowborder" valign="top" width="13.94%" headers="mcps1.2.4.1.1 "><p id="p151479166327"><a name="p151479166327"></a><a name="p151479166327"></a>offset</p>
</td>
<td class="cellrowborder" valign="top" width="12.98%" headers="mcps1.2.4.1.2 "><p id="p158449584436"><a name="p158449584436"></a><a name="p158449584436"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.08%" headers="mcps1.2.4.1.3 "><p id="p126017529210"><a name="p126017529210"></a><a name="p126017529210"></a>偏移offset个元素。</p>
</td>
</tr>
<tr id="row16103112474710"><td class="cellrowborder" valign="top" width="13.94%" headers="mcps1.2.4.1.1 "><p id="p20103424124712"><a name="p20103424124712"></a><a name="p20103424124712"></a>value</p>
</td>
<td class="cellrowborder" valign="top" width="12.98%" headers="mcps1.2.4.1.2 "><p id="p1810310245476"><a name="p1810310245476"></a><a name="p1810310245476"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.08%" headers="mcps1.2.4.1.3 "><p id="p5700191124716"><a name="p5700191124716"></a><a name="p5700191124716"></a>设置值。PrimType类型。</p>
<p id="p16103102414471"><a name="p16103102414471"></a><a name="p16103102414471"></a><span id="ph169021373405"><a name="ph169021373405"></a><a name="ph169021373405"></a>PrimType定义如下：</span></p>
<a name="zh-cn_topic_0000001491300625_screen115348581718"></a><a name="zh-cn_topic_0000001491300625_screen115348581718"></a><pre class="screen" codetype="Cpp" id="zh-cn_topic_0000001491300625_screen115348581718">// PrimT用于从T中提取基础数据类型：T传入基础数据类型，直接返回数据类型；T传入为TensorTrait类型时萃取TensorTrait中的LiteType基础数据类型
using PrimType = PrimT&lt;T&gt;;</pre>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无。

## 约束说明<a name="section633mcpsimp"></a>

无。

## 调用示例<a name="section17531157161314"></a>

```
// aGlobal数据 [1, 1, 1, 1, 1]
AscendC::GlobalTensor<T> aGlobal;
T val = 0;
aGlobal.SetValue(2, val);

//结果如下
aGlobal数据 [1, 1, 0, 1, 1]
```

