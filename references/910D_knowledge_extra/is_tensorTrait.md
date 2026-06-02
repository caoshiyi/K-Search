# is\_tensorTrait<a name="ZH-CN_TOPIC_0000002523304580"></a>

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
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p964420354242"><a name="p964420354242"></a><a name="p964420354242"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="zh-cn_topic_0000002078486173_zh-cn_topic_0000001576727153_zh-cn_topic_0000001389787297_section36583473819"></a>

判断输入的数据结构是否为TensorTrait数据结构，可通过检查其成员常量value的值来判断。当value为true时，表示输入的数据结构是TensorTrait类型；反之则为非TensorTrait类型。

## 函数原型<a name="zh-cn_topic_0000002078486173_zh-cn_topic_0000001576727153_zh-cn_topic_0000001389787297_section13230182415108"></a>

```
template <typename T> struct is_tensorTrait
```

## 参数说明<a name="zh-cn_topic_0000002078486173_zh-cn_topic_0000001576727153_zh-cn_topic_0000001389787297_section75395119104"></a>

<a name="zh-cn_topic_0000002078486173_zh-cn_topic_0000001576727153_zh-cn_topic_0000001389787297_table111938719446"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000002078486173_zh-cn_topic_0000001576727153_zh-cn_topic_0000001389787297_row6223476444"><th class="cellrowborder" valign="top" width="20.34%" id="mcps1.1.3.1.1"><p id="p1085176175119"><a name="p1085176175119"></a><a name="p1085176175119"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="79.66%" id="mcps1.1.3.1.2"><p id="p148519610515"><a name="p148519610515"></a><a name="p148519610515"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row1392614743211"><td class="cellrowborder" valign="top" width="20.34%" headers="mcps1.1.3.1.1 "><p id="p17843141016336"><a name="p17843141016336"></a><a name="p17843141016336"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="79.66%" headers="mcps1.1.3.1.2 "><p id="p1192718714328"><a name="p1192718714328"></a><a name="p1192718714328"></a>根据输入的数据类型，判断是否为<span>TensorTrait</span>数据结构。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0000002078486173_zh-cn_topic_0000001576727153_zh-cn_topic_0000001389787297_section25791320141317"></a>

无

## 约束说明<a name="zh-cn_topic_0000002078486173_zh-cn_topic_0000001576727153_zh-cn_topic_0000001389787297_section19165124931511"></a>

无

## 调用示例<a name="zh-cn_topic_0000002078486173_zh-cn_topic_0000001576727153_zh-cn_topic_0000001389787297_section320753512363"></a>

```
// 以下示例为基于googletest的UT示例

AscendC::Shape<int,int,int> shape = AscendC::MakeShape(10, 20, 30);
AscendC::Stride<int,int,int> stride = AscendC::MakeStride(1, 100, 200);

auto layoutMake = AscendC::MakeLayout(shape, stride);    
auto tensorTraitMake = AscendC::MakeTensorTrait<float, AscendC::TPosition::VECIN>(layoutMake);

EXPECT_EQ(AscendC::is_tensorTrait<decltype(shape)>::value, false);
EXPECT_EQ(AscendC::is_tensorTrait<decltype(stride)>::value, false);
EXPECT_EQ(AscendC::is_tensorTrait<decltype(layoutMake)>::value, false);
EXPECT_EQ(AscendC::is_tensorTrait<decltype(tensorTraitMake)>::value, true);
```

