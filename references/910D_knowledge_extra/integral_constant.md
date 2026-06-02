# integral\_constant<a name="ZH-CN_TOPIC_0000002523304870"></a>

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

integral\_constant是一个带有模板参数的结构体，定义在<type\_traits\>头文件中，用于封装一个编译时常量整数值，是标准库中许多类型特性和编译时计算的基础组件。

integral\_constant的功能如下：

1.  封装编译时常量，将一个int或bool类型的值封装为特定类型，以便该值可以在编译时被操作和传递。
2.  类型标识，每个不同的integral\_constant实例都是唯一的类型，可用于模板特化或重载决议。
3.  隐式转换时支持转换为其封装的值类型，便于在需要该值的上下文中直接使用。
4.  函数调用运算符允许像调用函数一样调用实例，以获取其值。

integral\_constant提供了多个常用的特化版本，具体如下：

-   Std::true\_type：integral\_constant<bool, true\>的别名。
-   Std::false\_type：integral\_constant<bool, false\>的别名。
-   数值常量：如Std::integral\_constant<int, 42\>。
-   数值常量的简化写法，Std::Int，数值类型为size\_t：如Std::Int<42\>。

## 函数原型<a name="section126881859101617"></a>

```
template <typename Tp, Tp v>
struct integral_constant
{
    static constexpr const Tp value = v;
    using value_type = Tp;
    using type = integral_constant;
    inline constexpr operator value_type() const noexcept;
    inline constexpr value_type operator()() const noexcept;
};

template <size_t v>
using Int = integral_constant<size_t, v>;
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
<td class="cellrowborder" valign="top" width="80.82000000000001%" headers="mcps1.2.3.1.2 "><p id="p366123884018"><a name="p366123884018"></a><a name="p366123884018"></a>数值的数据类型。</p>
</td>
</tr>
<tr id="row12299165018421"><td class="cellrowborder" valign="top" width="19.18%" headers="mcps1.2.3.1.1 "><p id="p3660123824013"><a name="p3660123824013"></a><a name="p3660123824013"></a>v</p>
</td>
<td class="cellrowborder" valign="top" width="80.82000000000001%" headers="mcps1.2.3.1.2 "><p id="p19788592347"><a name="p19788592347"></a><a name="p19788592347"></a>常量数值。</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section1564510486314"></a>

-   Int类型是integral\_constant数值结构的别名简写，数值类型必须是size\_t类型。
-   模板参数Tp不支持float等浮点数类型，因为模板参数需在编译期确定，而浮点数的精度问题可能导致编译期无法准确表示。

## 返回值说明<a name="section62431148556"></a>

无

## 调用示例<a name="section1193764916212"></a>

-   数值类型封装

    ```
    // 以下示例为基于googletest的UT示例
    using IntTrue = AscendC::Std::integral_constant<int, 1>;
    using IntFalse = AscendC::Std::integral_constant<int, 0>;
    // 测试 value 静态常量
    EXPECT_EQ(IntTrue::value, 1);
    EXPECT_EQ(IntFalse::value, 0);
    // 测试()操作符重载
    EXPECT_EQ(IntTrue()(), 1);
    EXPECT_EQ(IntFalse()(), 0);
    // 测试类型定义
    EXPECT_TRUE((AscendC::Std::is_same<typename IntTrue::value_type, int>::value));
    EXPECT_TRUE((AscendC::Std::is_same<typename IntTrue::type, IntTrue>::value));
    ```

-   特化类型——bool

    ```
    // 以下示例为基于googletest的UT示例
    using TrueType = AscendC::Std::true_type;
    using FalseType = AscendC::Std::false_type;
    // 测试 value 静态常量
    EXPECT_TRUE(TrueType::value);
    EXPECT_FALSE(FalseType::value);
    // 测试()操作符重载
    EXPECT_TRUE(TrueType()());
    EXPECT_FALSE(FalseType()());
    // 测试类型定义
    EXPECT_TRUE((AscendC::Std::is_same<typename TrueType::value_type, bool>::value));
    EXPECT_TRUE((AscendC::Std::is_same<typename TrueType::type, TrueType>::value));
    EXPECT_TRUE((AscendC::Std::is_same<typename FalseType::type, FalseType>::value));
    ```

-   特化类型——Int

    ```
    // 以下示例为基于googletest的UT示例
    using Zero = AscendC::Std::Int<0>;
    using One = AscendC::Std::Int<1>;
    using Large = AscendC::Std::Int<0xFFFFFFFF>;
    // 验证 value 静态常量
    EXPECT_EQ(Zero::value, 0);
    EXPECT_EQ(One::value, 1);
    EXPECT_EQ(Large::value, 0xFFFFFFFF);
    // 验证类型定义
    EXPECT_TRUE((AscendC::Std::is_same<typename Zero::value_type, size_t>::value));
    EXPECT_TRUE((AscendC::Std::is_same<typename Zero::type, Zero>::value));
    EXPECT_TRUE((AscendC::Std::is_same<Zero, AscendC::Std::integral_constant<size_t, 0>>::value));
    // 验证()操作符重载
    EXPECT_EQ(Zero()(), 0);
    EXPECT_EQ(One()(), 1);
    EXPECT_EQ(Large()(), 0xFFFFFFFF);
    ```

-   Int特化类型的运算

    ```
    // 加法
    static_assert((AscendC::Std::Int<5>::value + AscendC::Std::Int<3>::value) == 8, "Addition failed");
    // 乘法
    static_assert((AscendC::Std::Int<4>::value * AscendC::Std::Int<6>::value) == 24, "Multiplication failed");
    // 比较
    static_assert(AscendC::Std::Int<10>::value > AscendC::Std::Int<5>::value, "Comparison failed");
    static_assert(AscendC::Std::Int<7>::value != AscendC::Std::Int<77>::value, "Equality check failed");
    ```

