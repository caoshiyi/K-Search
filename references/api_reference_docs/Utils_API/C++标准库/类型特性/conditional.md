# conditional

**页面ID:** atlasascendc_api_07_10118  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_10118.html

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

conditional是定义在<type_traits>头文件里的一个类型特征工具，它在程序编译时根据一个布尔条件从两个类型中选择一个类型。本接口可应用在模板元编程中，用于根据不同的条件来灵活选择合适的类型，增强代码的通用性和灵活性。

conditional有一个嵌套的type成员，它的值取决于Bp的值：如果Bp为true，则conditional<Bp, If, Then>::type为If。如果Bp为false，则conditional<Bp, If, Then>::type为Then。

#### 函数原型

```
template <bool Bp, typename If, typename Then>
struct conditional;
```

#### 参数说明

**表1 **模板参数说明

| 参数名 | 含义 |
| --- | --- |
| Bp | 一个布尔常量表达式，作为选择类型的条件。 |
| If | 当Bp为true时选择的类型。 |
| Then | 当Bp为false时选择的类型。 |

#### 约束说明

无

#### 返回值说明

conditional的静态常量成员type用于获取返回值，conditional<Bp, If, Then>::type取值如下：

- If：Bp为true。
- Then：Bp为false。

#### 调用示例

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
