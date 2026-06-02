# is\_convertible<a name="ZH-CN_TOPIC_0000002554343993"></a>

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

is\_convertible是定义于<type\_traits\>头文件的一个类型转换检查工具，它提供了一种在程序编译时进行类型转换检查的机制：判断两个类型之间是否可以进行隐式转换并返回结果布尔值。本接口可应用在模板元编程、函数重载决议以及静态断言等场景，用于在程序编译阶段捕获潜在的类型转换错误，避免发生运行时错误。

## 函数原型<a name="section126881859101617"></a>

```
template <typename From, typename To>
struct is_convertible;
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
<tbody><tr id="row3502131221"><td class="cellrowborder" valign="top" width="19.18%" headers="mcps1.2.3.1.1 "><p id="p186611238204016"><a name="p186611238204016"></a><a name="p186611238204016"></a>From</p>
</td>
<td class="cellrowborder" valign="top" width="80.82000000000001%" headers="mcps1.2.3.1.2 "><p id="p366123884018"><a name="p366123884018"></a><a name="p366123884018"></a><span>源类型，即需要进行转换的原始类型。</span></p>
</td>
</tr>
<tr id="row12299165018421"><td class="cellrowborder" valign="top" width="19.18%" headers="mcps1.2.3.1.1 "><p id="p3660123824013"><a name="p3660123824013"></a><a name="p3660123824013"></a>To</p>
</td>
<td class="cellrowborder" valign="top" width="80.82000000000001%" headers="mcps1.2.3.1.2 "><p id="p126606385400"><a name="p126606385400"></a><a name="p126606385400"></a><span>目标类型，即</span><span>需要转换</span><span>到的目标类型。</span></p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section1564510486314"></a>

源类型和目标类型均不支持抽象类和多态类型。

## 返回值说明<a name="section62431148556"></a>

is\_convertible的静态常量成员value用于获取返回的布尔值，is\_convertible<From, To\>::value取值如下：

-   true：From类型的对象可以隐式转换为To类型。
-   false：From类型的对象不能隐式转换为To类型。

## 调用示例<a name="section1193764916212"></a>

```
class Base {};
class Derived : public Base {};
class Unrelated {};

// 检查 int 是否可以隐式转换为 double
AscendC::PRINTF("Is int convertible to double? %d\n", AscendC::Std::is_convertible<int, double>::value);

// 检查 double 是否可以隐式转换为 int
AscendC::PRINTF("Is double convertible to int? %d\n", AscendC::Std::is_convertible<double, int>::value);

// 检查 Derived 是否可以转换为 Base
AscendC::PRINTF("Is Derived callable with Base? %d\n", AscendC::Std::is_convertible<Derived, Base>::value);
// 检查 Base 是否可以转换为 Derived
AscendC::PRINTF("Is Base callable with Derived? %d\n", AscendC::Std::is_convertible<Base, Derived>::value);
// 检查 Derived 是否可以转换为 Unrelated
AscendC::PRINTF("Is Derived callable with Unrelated? %d\n", AscendC::Std::is_convertible<Derived, Unrelated>::value);
```

```
// 执行结果：
Is int convertible to double? 1
Is double convertible to int? 1
Is Derived callable with Base? 1
Is Base callable with Derived? 0
Is Derived callable with Unrelated? 0
```

