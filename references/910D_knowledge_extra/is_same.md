# is\_same<a name="ZH-CN_TOPIC_0000002523303604"></a>

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

is\_same是定义在<type\_traits\>头文件里的一个类型特征工具，它能够在程序编译时判断两个类型是否完全相同。本接口可应用在模板元编程、类型检查、条件编译等场景，用于在编译阶段确定类型信息，避免运行时可能出现的类型不匹配问题。

## 函数原型<a name="section126881859101617"></a>

```
template <typename Tp, typename Up>
struct is_same;
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
<td class="cellrowborder" valign="top" width="80.82000000000001%" headers="mcps1.2.3.1.2 "><p id="p366123884018"><a name="p366123884018"></a><a name="p366123884018"></a><span>需要比较</span><span>两个类型是否完全相同</span><span>的第一个类型。</span></p>
</td>
</tr>
<tr id="row12299165018421"><td class="cellrowborder" valign="top" width="19.18%" headers="mcps1.2.3.1.1 "><p id="p3660123824013"><a name="p3660123824013"></a><a name="p3660123824013"></a>Up</p>
</td>
<td class="cellrowborder" valign="top" width="80.82000000000001%" headers="mcps1.2.3.1.2 "><p id="p126606385400"><a name="p126606385400"></a><a name="p126606385400"></a><span>需要比较</span><span>两个类型是否完全相同</span><span>的第二个类型。</span></p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section1564510486314"></a>

无

## 返回值说明<a name="section62431148556"></a>

is\_same的静态常量成员value用于获取返回的布尔值，is\_same<Tp, Up\>::value取值如下：

-   true：Tp和Up是完全相同的类型。
-   false：Tp和Up不是相同的类型。

## 调用示例<a name="section1193764916212"></a>

```
// 定义两个不同的类
class ClassA {};
class ClassB {};

// 定义相同的类两次
class ClassC {};
using ClassC_alias = ClassC;

// 定义一个简单的模板类
template <typename T>
class TemplateClass {};

// 比较相同的基本类型
AscendC::PRINTF("Is int the same as int? %d\n", AscendC::Std::is_same<int, int>::value);

// 比较不同的基本类型
AscendC::PRINTF("Is int the same as double? %d\n", AscendC::Std::is_same<int, double>::value);

// 比较不同的类类型
AscendC::PRINTF("Is ClassA the same as ClassB? %d\n", AscendC::Std::is_same<ClassA, ClassB>::value);

// 比较相同的类类型
AscendC::PRINTF("Is ClassC the same as ClassC_alias? %d\n", AscendC::Std::is_same<ClassC, ClassC_alias>::value);

// 比较相同模板实例化类型
AscendC::PRINTF("Is TemplateClass<int> the same as TemplateClass<int>? %d\n", AscendC::Std::is_same<TemplateClass<int>, TemplateClass<int>>::value);

// 比较不同模板实例化类型
AscendC::PRINTF("Is TemplateClass<int> the same as TemplateClass<double>? %d\n", AscendC::Std::is_same<TemplateClass<int>, TemplateClass<double>>::value);
```

```
// 执行结果：
Is int the same as int? 1
Is int the same as double? 0
Is ClassA the same as ClassB? 0
Is ClassC the same as ClassC_alias? 1
Is TemplateClass<int> the same as TemplateClass<int>? 1
Is TemplateClass<int> the same as TemplateClass<double>? 0
```

