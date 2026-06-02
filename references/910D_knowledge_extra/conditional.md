# conditional<a name="ZH-CN_TOPIC_0000002523344032"></a>

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

conditional是定义在<type\_traits\>头文件里的一个类型特征工具，它在程序编译时根据一个布尔条件从两个类型中选择一个类型。本接口可应用在模板元编程中，用于根据不同的条件来灵活选择合适的类型，增强代码的通用性和灵活性。

conditional有一个嵌套的type成员，它的值取决于Bp的值：如果Bp为true，则conditional<Bp, If, Then\>::type为If。如果Bp为false，则conditional<Bp, If, Then\>::type为Then。

## 函数原型<a name="section126881859101617"></a>

```
template <bool Bp, typename If, typename Then>
struct conditional;
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
<tbody><tr id="row3502131221"><td class="cellrowborder" valign="top" width="19.18%" headers="mcps1.2.3.1.1 "><p id="p186611238204016"><a name="p186611238204016"></a><a name="p186611238204016"></a>Bp</p>
</td>
<td class="cellrowborder" valign="top" width="80.82000000000001%" headers="mcps1.2.3.1.2 "><p id="p366123884018"><a name="p366123884018"></a><a name="p366123884018"></a><span>一个布尔常量表达式，作为选择类型的条件。</span></p>
</td>
</tr>
<tr id="row12299165018421"><td class="cellrowborder" valign="top" width="19.18%" headers="mcps1.2.3.1.1 "><p id="p3660123824013"><a name="p3660123824013"></a><a name="p3660123824013"></a>If</p>
</td>
<td class="cellrowborder" valign="top" width="80.82000000000001%" headers="mcps1.2.3.1.2 "><p id="p126606385400"><a name="p126606385400"></a><a name="p126606385400"></a><span>当</span><span>Bp</span><span>为</span><span>true</span><span>时选择的类型</span><span>。</span></p>
</td>
</tr>
<tr id="row17794174201710"><td class="cellrowborder" valign="top" width="19.18%" headers="mcps1.2.3.1.1 "><p id="p87944411710"><a name="p87944411710"></a><a name="p87944411710"></a>Then</p>
</td>
<td class="cellrowborder" valign="top" width="80.82000000000001%" headers="mcps1.2.3.1.2 "><p id="p979414418175"><a name="p979414418175"></a><a name="p979414418175"></a><span>当</span>B<span>p为</span>false<span>时选择的类型</span>。</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section1564510486314"></a>

无

## 返回值说明<a name="section62431148556"></a>

conditional的静态常量成员type用于获取返回值，conditional<Bp, If, Then\>::type取值如下：

-   If：Bp为true。
-   Then：Bp为false。

## 调用示例<a name="section1193764916212"></a>

```
// 定义两个不同的类型
struct TypeA {
    __aicore__ inline static void print() {
        AscendC::PRINTF("This is TypeA..\n");
    }
};

struct TypeB {
    __aicore__ inline static void print() {
        AscendC::PRINTF("This is TypeB..\n");
    }
};

// 根据条件选择类型
template <bool Condition>
__aicore__ inline void selectType() {
    using SelectedType = typename AscendC::Std::conditional<Condition, TypeA, TypeB>::type;
    SelectedType::print();
}

// 定义一个模板函数，根据条件选择不同的类型
template <bool Condition>
__aicore__ inline void selectOtherType() {
    using SelectedType = typename std::conditional<Condition, int, float>::type;
    if constexpr (std::is_same_v<SelectedType, int>) {
        AscendC::PRINTF("Selected type is int.\n");
    } else {
        AscendC::PRINTF("Selected type is float.\n");
    }
}

// 条件为 true，选择 TypeA
selectType<true>();
// 条件为 false，选择 TypeB
selectType<false>();

// 测试条件为 true 的情况
selectOtherType<true>();
// 测试条件为 false 的情况
selectOtherType<false>();
```

```
// 执行结果：
This is TypeA..
This is TypeB..
Selected type is int.
Selected type is float.
```

