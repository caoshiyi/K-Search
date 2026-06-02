# integral_constant

**页面ID:** atlasascendc_api_07_10198  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_10198.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | x |
| Atlas 推理系列产品AI Core | x |
| Atlas 推理系列产品Vector Core | x |
| Atlas 训练系列产品 | x |

#### 功能说明

integral_constant是一个带有模板参数的结构体，定义在<type_traits>头文件中，用于封装一个编译时常量整数值，是标准库中许多类型特性和编译时计算的基础组件。

integral_constant的功能如下：

1. 封装编译时常量，将一个int或bool类型的值封装为特定类型，以便该值可以在编译时被操作和传递。
2. 类型标识，每个不同的integral_constant实例都是唯一的类型，可用于模板特化或重载决议。
3. 隐式转换时支持转换为其封装的值类型，便于在需要该值的上下文中直接使用。
4. 函数调用运算符允许像调用函数一样调用实例，以获取其值。

integral_constant提供了多个常用的特化版本，具体如下：

- Std::true_type：integral_constant<bool, true>的别名。
- Std::false_type：integral_constant<bool, false>的别名。
- 数值常量：如Std::integral_constant<int, 42>。
- 数值常量的简化写法，Std::Int，数值类型为size_t：如Std::Int<42>。

#### 函数原型

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

#### 参数说明

**表1 **模板参数说明

| 参数名 | 含义 |
| --- | --- |
| Tp | 数值的数据类型。 |
| v | 常量数值。 |

#### 约束说明

- Int类型是integral_constant数值结构的别名简写，数值类型必须是size_t类型。
- 模板参数Tp不支持float等浮点数类型，因为模板参数需在编译期确定，而浮点数的精度问题可能导致编译期无法准确表示。

#### 调用示例

- 数值类型封装

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

- 特化类型——bool

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

- 特化类型——Int

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

- Int特化类型的运算

```
// 加法
static_assert((AscendC::Std::Int<5>::value + AscendC::Std::Int<3>::value) == 8, "Addition failed");
// 乘法
static_assert((AscendC::Std::Int<4>::value * AscendC::Std::Int<6>::value) == 24, "Multiplication failed");
// 比较
static_assert(AscendC::Std::Int<10>::value > AscendC::Std::Int<5>::value, "Comparison failed");
static_assert(AscendC::Std::Int<7>::value != AscendC::Std::Int<77>::value, "Equality check failed");
```
