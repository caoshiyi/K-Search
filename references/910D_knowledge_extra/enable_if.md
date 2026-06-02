# enable\_if<a name="ZH-CN_TOPIC_0000002554423485"></a>

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

enable\_if是定义于<type\_traits\>头文件的一个模板元编程工具，它能够在程序编译时根据某个条件启用或禁用特定的函数模板、类模板或模板特化，以此实现更精细的模板重载和类型选择，增强代码的灵活性和安全性。

enable\_if是一个模板结构体，有两个模板参数：模板参数Bp是一个布尔值，表示条件；模板参数Tp是一个类型，默认值为void。当Bp为false时，enable\_if没有嵌套的type成员。当Bp为true时，enable\_if有一个嵌套的type成员，其类型为Tp。

## 函数原型<a name="section126881859101617"></a>

```
template <bool Bp, typename Tp>
struct enable_if;
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
<td class="cellrowborder" valign="top" width="80.82000000000001%" headers="mcps1.2.3.1.2 "><p id="p366123884018"><a name="p366123884018"></a><a name="p366123884018"></a>布尔值，表示条件。</p>
</td>
</tr>
<tr id="row12299165018421"><td class="cellrowborder" valign="top" width="19.18%" headers="mcps1.2.3.1.1 "><p id="p3660123824013"><a name="p3660123824013"></a><a name="p3660123824013"></a>Tp</p>
</td>
<td class="cellrowborder" valign="top" width="80.82000000000001%" headers="mcps1.2.3.1.2 "><p id="p126606385400"><a name="p126606385400"></a><a name="p126606385400"></a>类型，默认值为void。</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section1564510486314"></a>

无

## 返回值说明<a name="section62431148556"></a>

enable\_if的静态常量成员type用于获取返回值，enable\_if<Bp,  Tp\>::type取值如下：

-   Tp：Bp为true。
-   void：Bp为false。

## 调用示例<a name="section1193764916212"></a>

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

