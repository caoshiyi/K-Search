# is_convertible

**页面ID:** atlasascendc_api_07_10114  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_10114.html

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

is_convertible是定义于<type_traits>头文件的一个类型转换检查工具，它提供了一种在程序编译时进行类型转换检查的机制：判断两个类型之间是否可以进行隐式转换并返回结果布尔值。本接口可应用在模板元编程、函数重载决议以及静态断言等场景，用于在程序编译阶段捕获潜在的类型转换错误，避免发生运行时错误。

#### 函数原型

```
template <typename From, typename To>
struct is_convertible;
```

#### 参数说明

**表1 **模板参数说明

| 参数名 | 含义 |
| --- | --- |
| From | 源类型，即需要进行转换的原始类型。 |
| To | 目标类型，即需要转换到的目标类型。 |

#### 约束说明

源类型和目标类型均不支持抽象类和多态类型。

#### 返回值说明

is_convertible的静态常量成员value用于获取返回的布尔值，is_convertible<From, To>::value取值如下：

- true：From类型的对象可以隐式转换为To类型。
- false：From类型的对象不能隐式转换为To类型。

#### 调用示例

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
