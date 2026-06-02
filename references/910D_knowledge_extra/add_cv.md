# add\_cv<a name="ZH-CN_TOPIC_0000002554344813"></a>

## 产品支持情况<a name="section1586581915393"></a>

<a name="table169596713360"></a>
<table><thead align="left"><tr id="row129590715369"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p17959971362"><a name="p17959971362"></a><a name="p17959971362"></a><span id="ph895914718367"><a name="ph895914718367"></a><a name="ph895914718367"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p89594763612"><a name="p89594763612"></a><a name="p89594763612"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row18959673369"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p1595910763613"><a name="p1595910763613"></a><a name="p1595910763613"></a><span id="ph1595918753613"><a name="ph1595918753613"></a><a name="ph1595918753613"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p1695957133611"><a name="p1695957133611"></a><a name="p1695957133611"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section7376114729"></a>

在程序编译时，为指定类型添加const和volatile限定符，可以用于在编译时进行类型转换。

## 函数原型<a name="section126881859101617"></a>

```
template <typename Tp>
struct add_cv;
```

## 参数说明<a name="section121562129312"></a>

**表 1**  模板参数说明

<a name="table729818506422"></a>
<table><thead align="left"><tr id="row11299950204217"><th class="cellrowborder" valign="top" width="19.18%" id="mcps1.2.3.1.1"><p id="p1029955044218"><a name="p1029955044218"></a><a name="p1029955044218"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="80.82000000000001%" id="mcps1.2.3.1.2"><p id="p1121663111288"><a name="p1121663111288"></a><a name="p1121663111288"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row3502131221"><td class="cellrowborder" valign="top" width="19.18%" headers="mcps1.2.3.1.1 "><p id="p186611238204016"><a name="p186611238204016"></a><a name="p186611238204016"></a>Tp</p>
</td>
<td class="cellrowborder" valign="top" width="80.82000000000001%" headers="mcps1.2.3.1.2 "><p id="p366123884018"><a name="p366123884018"></a><a name="p366123884018"></a><span>需要处理的类型</span>，<span>包括基本类型（如</span>int<span>、</span>float<span>等）、复合类型（如数组、指针、引用）、用户自定义类型（如类、结构体等），以及带有</span>const<span>限定符、</span>volatile<span>限定符</span><span>或这两个限定符的类型。</span></p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section1564510486314"></a>

无

## 返回值说明<a name="section62431148556"></a>

add\_cv是一个结构体，其提供一个嵌套类型type，表示添加const和volatile限定符后的类型。通过add\_cv<Tp\>::type来访问该类型。

## 调用示例<a name="section1193764916212"></a>

```
// Test non-const and non-volatile type
ascendc_assert((AscendC::Std::is_same_v<AscendC::Std::add_cv<int>::type, const volatile int>));
ascendc_assert((AscendC::Std::is_same_v<AscendC::Std::add_cv<double>::type, const volatile double>));
ascendc_assert((AscendC::Std::is_same_v<AscendC::Std::add_cv<char>::type, const volatile char>));
// Test const type
ascendc_assert((AscendC::Std::is_same_v<AscendC::Std::add_cv<const int>::type, const volatile int>));
ascendc_assert((AscendC::Std::is_same_v<AscendC::Std::add_cv<const double>::type, const volatile double>));
ascendc_assert((AscendC::Std::is_same_v<AscendC::Std::add_cv<const char>::type, const volatile char>));
// Test volatile type
ascendc_assert((AscendC::Std::is_same_v<AscendC::Std::add_cv<volatile int>::type, const volatile int>));
ascendc_assert((AscendC::Std::is_same_v<AscendC::Std::add_cv<volatile double>::type, const volatile double>));
ascendc_assert((AscendC::Std::is_same_v<AscendC::Std::add_cv<volatile char>::type, const volatile char>));
// Test const and volatile type
ascendc_assert((AscendC::Std::is_same_v<AscendC::Std::add_cv<const volatile int>::type, const volatile int>));
ascendc_assert((AscendC::Std::is_same_v<AscendC::Std::add_cv<const volatile double>::type, const volatile double>));
ascendc_assert((AscendC::Std::is_same_v<AscendC::Std::add_cv<const volatile char>::type, const volatile char>));

ascendc_assert((AscendC::Std::is_same_v<AscendC::Std::add_cv_t<int>, const volatile int>));
ascendc_assert((AscendC::Std::is_same_v<AscendC::Std::add_cv_t<double>, const volatile double>));

ascendc_assert((AscendC::Std::is_same_v<AscendC::Std::add_cv_t<const int>, const volatile int>));
ascendc_assert((AscendC::Std::is_same_v<AscendC::Std::add_cv_t<const double>, const volatile double>));

ascendc_assert((AscendC::Std::is_same_v<AscendC::Std::add_cv_t<volatile int>, const volatile int>));
ascendc_assert((AscendC::Std::is_same_v<AscendC::Std::add_cv_t<volatile double>, const volatile double>));

ascendc_assert((AscendC::Std::is_same_v<AscendC::Std::add_cv_t<const volatile int>, const volatile int>));
ascendc_assert((AscendC::Std::is_same_v<AscendC::Std::add_cv_t<const volatile double>, const volatile double>));
```

