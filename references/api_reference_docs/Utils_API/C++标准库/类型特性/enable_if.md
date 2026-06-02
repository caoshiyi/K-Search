# enable_if

**页面ID:** atlasascendc_api_07_10117  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_10117.html

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

enable_if是定义于<type_traits>头文件的一个模板元编程工具，它能够在程序编译时根据某个条件启用或禁用特定的函数模板、类模板或模板特化，以此实现更精细的模板重载和类型选择，增强代码的灵活性和安全性。

enable_if是一个模板结构体，有两个模板参数：模板参数Bp是一个布尔值，表示条件；模板参数Tp是一个类型，默认值为void。当Bp为false时，enable_if没有嵌套的type成员。当Bp为true时，enable_if有一个嵌套的type成员，其类型为Tp。

#### 函数原型

```
template <bool Bp, typename Tp>
struct enable_if;
```

#### 参数说明

**表1 **模板参数说明

| 参数名 | 含义 |
| --- | --- |
| Bp | 布尔值，表示条件。 |
| Tp | 类型，默认值为void。 |

#### 约束说明

无

#### 返回值说明

enable_if的静态常量成员type用于获取返回值，enable_if<Bp,  Tp>::type取值如下：

- Tp：Bp为true。
- void：Bp为false。

#### 调用示例

```
template <typename T>
class Calculator {
public:
    // 当 T 是整数类型时启用此成员函数
    template <typename U = T>
    typename AscendC::Std::enable_if<AscendC::Std::is_integral<U>::value, U>::type
    __aicore__ inline multiply(U a, U b) {
        AscendC::PRINTF("Integral type multiplication");
        return a * b;
    }

    // 当 T 不是整数类型时启用此成员函数
    template <typename U = T>
    typename AscendC::Std::enable_if<!AscendC::Std::is_integral<U>::value, U>::type
    __aicore__ inline multiply(U a, U b) {
        AscendC::PRINTF("Non-integral type multiplication");
        return a * b;
    }
};

// 通用模板类
template <typename T, typename Enable = void>
class Container {
public:
    __aicore__ inline Container() {
        AscendC::PRINTF("Generic container.\n");
    }
};

// 特化版本，当 T 是整数类型时启用
template <typename T>
class Container<T, typename AscendC::Std::enable_if<AscendC::Std::is_integral<T>::value>::type> {
public:
    __aicore__ inline Container() {
        AscendC::PRINTF("Integral container.\n");
    }
};

// 当 T 是整数类型时启用该函数
template <typename T> 
__aicore__ inline typename AscendC::Std::enable_if<AscendC::Std::is_integral<T>::value, T>::type add(T a, T b) {
    AscendC::PRINTF("Integral type addition.");
    return a + b;
}

// 当 T 不是整数类型时启用该函数
template <typename T> 
__aicore__ inline typename AscendC::Std::enable_if<!AscendC::Std::is_integral<T>::value, T>::type add(T a, T b) {
    AscendC::PRINTF("Non-integral type addition.");
    return a + (-b);
}

Calculator<int> intCalculator;
int intResult = intCalculator.multiply((int)2, (int)3);
AscendC::PRINTF("Result of integral multiplication: %d\n", intResult);

Calculator<float> doubleCalculator;
float doubleResult = doubleCalculator.multiply((float)2.5, (float)3.5);
AscendC::PRINTF("Result of non-integral multiplication: %f\n", doubleResult);

Container<float> genericContainer;
Container<int> integralContainer;

intResult = add(1, 2);
AscendC::PRINTF("Integer result: %d\n", intResult);

doubleResult = add((float)1.5, (float)2.5);
AscendC::PRINTF("float result: %f\n", doubleResult);
```

```
// 执行结果：
Integral type multiplicationResult of integral multiplication: 6
Non-integral type multiplicationResult of non-integral multiplication: 8.750000
Generic container.
Integral container.
Integral type addition.Integer result: 3
Non-integral type addition.float result: -1.000000
```
