# is\_base\_of<a name="ZH-CN_TOPIC_0000002523343810"></a>

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

is\_base\_of是定义于<type\_traits\>头文件的一个类型特征工具，它能够在程序编译时检查一个类型是否为另一个类型的基类。本接口可应用在模板元编程、类型检查和条件编译等场景，用于在编译阶段捕获潜在的类型错误，提高代码的鲁棒性。

## 函数原型<a name="section126881859101617"></a>

```
template <typename Base, typename Derived>
struct is_base_of;
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
<tbody><tr id="row3502131221"><td class="cellrowborder" valign="top" width="19.18%" headers="mcps1.2.3.1.1 "><p id="p186611238204016"><a name="p186611238204016"></a><a name="p186611238204016"></a>Base</p>
</td>
<td class="cellrowborder" valign="top" width="80.82000000000001%" headers="mcps1.2.3.1.2 "><p id="p366123884018"><a name="p366123884018"></a><a name="p366123884018"></a><span>待检查的基类类型，即Base类型是否为Derived类型的</span><span>基类</span><span>。</span></p>
</td>
</tr>
<tr id="row12299165018421"><td class="cellrowborder" valign="top" width="19.18%" headers="mcps1.2.3.1.1 "><p id="p3660123824013"><a name="p3660123824013"></a><a name="p3660123824013"></a>Derived</p>
</td>
<td class="cellrowborder" valign="top" width="80.82000000000001%" headers="mcps1.2.3.1.2 "><p id="p126606385400"><a name="p126606385400"></a><a name="p126606385400"></a><span>待检查的派生类类型</span><span>，即Base类型是否为Derived类型的</span><span>基类</span><span>。</span></p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section1564510486314"></a>

无

## 返回值说明<a name="section62431148556"></a>

is\_base\_of的静态常量成员value用于获取返回的布尔值，is\_base\_of<Base, Derived\>::value取值如下：

-   true：Base类型是Derived类型的基类（包括Base类型和Derived类型为同一类型的情况）。
-   false：Base类型不是Derived类型的基类。

## 调用示例<a name="section1193764916212"></a>

```
class Base {};
class Derived : public Base {};
class Unrelated {};

// 虚继承的派生类
class Derived2 : virtual public Base {};

// 定义虚继承的派生类
class VirtualDerived : virtual public Base {};

// 定义多重继承的派生类
class MultiDerived : public Base, public VirtualDerived {};

// 模板基类
template <typename T>
class BaseTemplate {
public:
    T value;
};

// 模板派生类
template <typename T>
class DerivedTemplate : public BaseTemplate<T> {};

// 检查 Base 是否是 Derived 的基类
AscendC::PRINTF("Is Base a base of Derived? %d\n" , AscendC::Std::is_base_of<Base, Derived>::value);

// 检查 Derived 是否是 Base 的基类（应该为 false）
AscendC::PRINTF("Is Derived a base of Base? %d\n" , AscendC::Std::is_base_of<Derived, Base>::value);

// 检查 Base 是否是 Unrelated 的基类（应该为 false）
AscendC::PRINTF("Is Base a base of Unrelated? %d\n" , AscendC::Std::is_base_of<Base, Unrelated>::value);

AscendC::PRINTF("Is Base a base of Derived (virtual inheritance)? %d\n", AscendC::Std::is_base_of<Base, Derived2>::value);

AscendC::PRINTF("Is BaseTemplate<int> a base of DerivedTemplate<int>? %d\n", AscendC::Std::is_base_of<BaseTemplate<int>, DerivedTemplate<int>>::value);

// 测试 Base 是否为 VirtualDerived 的基类（虚继承情况）
AscendC::PRINTF("Is Base a base of VirtualDerived? %d\n" , AscendC::Std::is_base_of<Base, VirtualDerived>::value);
// 测试 Base 是否为 MultiDerived 的基类（多重继承情况）
AscendC::PRINTF("Is Base a base of MultiDerived? %d\n" , AscendC::Std::is_base_of<Base, MultiDerived>::value);
```

```
// 执行结果：
Is Base a base of Derived? 1
Is Derived a base of Base? 0
Is Base a base of Unrelated? 0
Is Base a base of Derived (virtual inheritance)? 1
Is BaseTemplate<int> a base of DerivedTemplate<int>? 1
Is Base a base of VirtualDerived? 1
Is Base a base of MultiDerived? 1
```

