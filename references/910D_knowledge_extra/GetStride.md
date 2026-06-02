# GetStride<a name="ZH-CN_TOPIC_0000002554424205"></a>

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
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p37256491200"><a name="p37256491200"></a><a name="p37256491200"></a>x</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="zh-cn_topic_0000002042526794_zh-cn_topic_0000001602767550_zh-cn_topic_0000001600123102_section36583473819"></a>

返回描述内存访问步长的Stride对象，与Shape的维度信息一一对应。

## 函数原型<a name="zh-cn_topic_0000002042526794_zh-cn_topic_0000001602767550_zh-cn_topic_0000001600123102_section13230182415108"></a>

```
__aicore__ inline constexpr decltype(auto) GetStride() {}    
__aicore__ inline constexpr decltype(auto) GetStride() const {}
```

## 参数说明<a name="zh-cn_topic_0000002042526794_zh-cn_topic_0000001602767550_zh-cn_topic_0000001600123102_section75395119104"></a>

无

## 返回值说明<a name="zh-cn_topic_0000002042526794_zh-cn_topic_0000001602767550_zh-cn_topic_0000001600123102_section25791320141317"></a>

描述内存访问步长的Stride对象，Stride结构类型（[Std::tuple](容器函数.md)类型的别名），定义如下：

```
template <typename... Strides>
using Stride = Std::tuple<Strides...>;
```

## 约束说明<a name="zh-cn_topic_0000002042526794_zh-cn_topic_0000001602767550_zh-cn_topic_0000001600123102_section19165124931511"></a>

无

## 调用示例<a name="zh-cn_topic_0000002042526794_zh-cn_topic_0000001602767550_zh-cn_topic_0000001600123102_section320753512363"></a>

```
// 初始化Layout数据结构，获取对应数值
AscendC::Shape<int,int,int> shape = AscendC::MakeShape(10, 20, 30);
AscendC::Stride<int,int,int> stride = AscendC::MakeStride(1, 100, 200);

auto layoutMake = AscendC::MakeLayout(shape, stride);
AscendC::Layout<AscendC::Shape<int, int, int>, AscendC::Stride<int, int, int>> layoutInit(shape, stride);

int value = AscendC::Std::get<0>(layoutInit.GetStride()); // value = 1
value = AscendC::Std::get<1>(layoutInit.GetStride()); // value = 100
value = AscendC::Std::get<2>(layoutInit.GetStride()); // value = 200
```

