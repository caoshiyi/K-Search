# complex32/complex64<a name="ZH-CN_TOPIC_0000002554424693"></a>

## 产品支持情况<a name="section73648168211"></a>

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

复数类型，其中complex32表示实部和虚部都是half类型的复数，位宽为32位；complex64表示实部和虚部都是float类型的复数，位宽为64位。

具体定义如下：

```
namespace AscendC {
template<class T>
struct Complex {
    T real;
    T imag;
};
} // namespace AscendC
using complex32 = AscendC::Complex<half>;
using complex64 = AscendC::Complex<float>;
```

**表 1**  Complex模板参数说明

<a name="zh-cn_topic_0000001441184464_table18149577913"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001441184464_row61411571196"><th class="cellrowborder" valign="top" width="19.59%" id="mcps1.2.3.1.1"><p id="zh-cn_topic_0000001441184464_p2093713281104"><a name="zh-cn_topic_0000001441184464_p2093713281104"></a><a name="zh-cn_topic_0000001441184464_p2093713281104"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="80.41%" id="mcps1.2.3.1.2"><p id="zh-cn_topic_0000001441184464_p393813285106"><a name="zh-cn_topic_0000001441184464_p393813285106"></a><a name="zh-cn_topic_0000001441184464_p393813285106"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001441184464_row8906103284616"><td class="cellrowborder" valign="top" width="19.59%" headers="mcps1.2.3.1.1 "><p id="p71771922134011"><a name="p71771922134011"></a><a name="p71771922134011"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="80.41%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001441184464_p14953134584410"><a name="zh-cn_topic_0000001441184464_p14953134584410"></a><a name="zh-cn_topic_0000001441184464_p14953134584410"></a>实部和虚部的数据类型，仅支持half/float。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  Complex结构体参数说明

<a name="table153364918102"></a>
<table><thead align="left"><tr id="row7363209171013"><th class="cellrowborder" valign="top" width="19.62%" id="mcps1.2.3.1.1"><p id="p136399171010"><a name="p136399171010"></a><a name="p136399171010"></a><strong id="b137544519107"><a name="b137544519107"></a><a name="b137544519107"></a>函数名称</strong></p>
</th>
<th class="cellrowborder" valign="top" width="80.38%" id="mcps1.2.3.1.2"><p id="p7363209141012"><a name="p7363209141012"></a><a name="p7363209141012"></a><strong id="b1767135119100"><a name="b1767135119100"></a><a name="b1767135119100"></a>入参说明</strong></p>
</th>
</tr>
</thead>
<tbody><tr id="row2036317913108"><td class="cellrowborder" valign="top" width="19.62%" headers="mcps1.2.3.1.1 "><p id="p57906662716"><a name="p57906662716"></a><a name="p57906662716"></a>real</p>
</td>
<td class="cellrowborder" valign="top" width="80.38%" headers="mcps1.2.3.1.2 "><p id="p5441241210"><a name="p5441241210"></a><a name="p5441241210"></a>实部，类型为T，仅支持half/float。</p>
</td>
</tr>
<tr id="row424115175561"><td class="cellrowborder" valign="top" width="19.62%" headers="mcps1.2.3.1.1 "><p id="p11241181710563"><a name="p11241181710563"></a><a name="p11241181710563"></a>imag</p>
</td>
<td class="cellrowborder" valign="top" width="80.38%" headers="mcps1.2.3.1.2 "><p id="p172416177566"><a name="p172416177566"></a><a name="p172416177566"></a>虚部，类型为T，仅支持half/float。</p>
</td>
</tr>
</tbody>
</table>

使用示例如下：

```
// value0代表实部为1，虚部为2的复数，即1+2j
complex32 value0(1, 2);
// value1代表实部为3，虚部为0的复数，即3+0j
complex32 value1(3);
// value2代表实部为4，虚部为0的复数，即4+0j
complex64 value2 = 4;
```

