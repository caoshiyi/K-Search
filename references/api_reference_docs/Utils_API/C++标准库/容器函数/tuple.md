# tuple

**页面ID:** atlasascendc_api_07_10108  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_10108.html

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

在C++中，tuple是一个功能强大的容器，它允许存储多个不同类型的元素。具体使用场景如下：

- 当一个函数需要返回多个不同类型的值时，使用tuple是一个很好的选择。它避免了创建结构体或类来封装多个返回值，尤其是在不需要为这些返回值创建单独的类型时。
- 当需要存储不同类型的数据，可以使用tuple来存储异构元素。例如，存储数据的查询结果，其中包含不同类型的字段。
- 在函数调用中，当需要传递多个不同类型的参数，但又不想为它们创建结构体时，可以使用tuple对这些参数分组。
- 在模板元编程中，tuple可以作为类型列表，存储多个类型信息。
- 当程序需要在不同阶段存储和传递多个状态信息时，可以使用tuple将这些状态统一存储，便于管理。
- 在一些泛型算法中，可以使用tuple存储计算结果或中间结果。
- 在模板元编程中，可以使用tuple存储和展开参数包。

以下是tuple的构造函数说明：

- 默认构造函数：创建一个空的元组。
- 初始化列表构造函数：直接在创建对象时提供元素的值。
- 复制构造函数：用来复制一个已有的元组。

另外，Ascend C提供辅助函数make_tuple，用于创建元组，它可以自动推断元素的类型，使代码更简洁，也可以使用make_tuple来构造元素列表。

#### 函数原型

```
template <typename Tp, typename ...Tps>
class tuple<Tp, Tps...> : public tuple<Tps...> 
{
public:
    __aicore__ inline tuple();

    __aicore__ inline tuple(const Tp& val, const Tps& ...params);

    __aicore__ inline tuple(Tp&& val, Tps&& ...params);

    template <typename Head, typename ...Args>
    __aicore__ inline tuple<Tp, Tps...>& operator=(const tuple<Head, Args...>& t);
}
```

#### 参数说明

**表1 **模板参数说明

| 参数名 | 含义 |
| --- | --- |
| Tps... | 表示输入类型的形参包，参数个数范围为[0，64]。 Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持的数据类型为：bool、int4b_t、int8_t、uint8_t、int16_t、uint16_t、half、bfloat16_t、int32_t、uint32_t、float、int64_t、uint64_t、LocalTensor、GlobalTensor。 Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持的数据类型为：bool、int4b_t、int8_t、uint8_t、int16_t、uint16_t、half、bfloat16_t、int32_t、uint32_t、float、int64_t、uint64_t、LocalTensor、GlobalTensor。 |

#### 约束说明

- tuple实例化深度为64，即支持64个元素以内的数据类型的聚合。
- 构造函数为初始化列表时，列表中的数据应同为左值或右值，不可混用，初始化列表中的元素个数和类型，必须与tuple定义的元素个数和类型严格匹配。
- 不支持数组等可变长度的数据类型。
- 不支持隐式转换构造函数。
- 由于当前基于继承实现tuple，对于如tuple<int, int> a =make_tuple(1, 2, 3)的写法，会触发隐式转换，同时因为C++ object slice数据切片的特性，会导致多余的部分数据被丢弃，因此，应避免使用上述写法。

#### 调用示例

```
AscendC::LocalTensor<T> src0Local = inQueueX.AllocTensor<T>();
AscendC::LocalTensor<T> src1Local = inQueueX2.AllocTensor<T>();

// make_tuple聚合Tensor类结构
auto testMakeTensor = AscendC::Std::make_tuple(src0Local, src1Local, src0_global, src1_global);

AscendC::PRINTF("tuple size is --> %d\n", AscendC::Std::tuple_size<decltype(testMakeTensor)>::value);

//初始化列表构造聚合
AscendC::Std::tuple<AscendC::LocalTensor<T>, AscendC::LocalTensor<T>, AscendC::GlobalTensor<T>, AscendC::GlobalTensor<T>> testTensor{src0Local, src1Local, src0_global, src1_global};

// 复制构造方式
AscendC::Std::tuple<AscendC::LocalTensor<T>, AscendC::LocalTensor<T>, AscendC::GlobalTensor<T>, AscendC::GlobalTensor<T>> test2Tensor = testTensor;

AscendC::Std::tuple<AscendC::LocalTensor<T>, AscendC::LocalTensor<T>, AscendC::GlobalTensor<T>, AscendC::GlobalTensor<T>> test3Tensor;

// 运算符重载
test3Tensor = test2Tensor;

AscendC::PRINTF("tuple size is --> %d\n", AscendC::Std::tuple_size<decltype(test3Tensor)>::value);

// get方法获取对应元素
AscendC::LocalTensor<T> src0LocalTuple = AscendC::Std::get<0>(test3Tensor);
AscendC::LocalTensor<T> src1LocalTuple = AscendC::Std::get<1>(test3Tensor);

AscendC::GlobalTensor<T> src0_globalTuple = AscendC::Std::get<2>(test3Tensor);
AscendC::GlobalTensor<T> src1_globalTuple = AscendC::Std::get<3>(test3Tensor);        

// 多种数据类型初始化列表聚合
AscendC::Std::tuple<AscendC::int4b_t, int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, uint64_t, int64_t, half, float, \
bfloat16_t, fp8_e8m0_t, bool> test1 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10.0, 11.0, 12.0, 13.0, true};

AscendC::PRINTF("tuple size is --> %d\n", AscendC::Std::tuple_size<decltype(test1)>::value);

// get方法获取多种数据类型元素
AscendC::int4b_t        number_int4b_t =        AscendC::Std::get<0>(test1);
int8_t                  number_int8_t =         AscendC::Std::get<1>(test1);
uint8_t                 number_uint8_t =        AscendC::Std::get<2>(test1);
int16_t                 number_int16_t =        AscendC::Std::get<3>(test1);
uint16_t                number_uint16_t =       AscendC::Std::get<4>(test1);
int32_t                 number_int32_t =        AscendC::Std::get<5>(test1);
uint32_t                number_uint32_t =       AscendC::Std::get<6>(test1);
uint64_t                number_uint64_t =       AscendC::Std::get<7>(test1);
int64_t                 number_int64_t =        AscendC::Std::get<8>(test1);
half                    number_half =           AscendC::Std::get<9>(test1);
float                   number_float =          AscendC::Std::get<10>(test1);
bfloat16_t              number_bfloat16_t =     AscendC::Std::get<11>(test1);
fp8_e8m0_t              number_fp8_e8m0_t =     AscendC::Std::get<12>(test1);
bool                    number_bool =           AscendC::Std::get<13>(test1);

// get方法获取元素引用接续运算
AscendC::Std::get<1>(test1)+= 1 ;
AscendC::Std::get<2>(test1)+= 1 ;
AscendC::Std::get<3>(test1)+= 1 ;
AscendC::Std::get<4>(test1)+= 1 ;
AscendC::Std::get<5>(test1)+= 1 ;
AscendC::Std::get<6>(test1)+= 1 ;
AscendC::Std::get<7>(test1)+= 1 ;
AscendC::Std::get<8>(test1)+= 1 ;
AscendC::Std::get<10>(test1) += (float)1.0 ;

// make_tuple初始化列表固定元素数据类型
auto test2 = AscendC::Std::make_tuple(AscendC::int4b_t (1) ,int8_t (2) ,uint8_t (3) ,int16_t (4) ,uint16_t (5) ,int32_t (6) , \
    uint32_t (7) ,uint64_t (8) ,int64_t (9) ,half (10) ,float (11) ,bfloat16_t (12) ,fp8_e8m0_t (13) , bool (true));

AscendC::PRINTF("tuple size is --> %d\n", AscendC::Std::tuple_size<decltype(test2)>::value);

// get方法获取多种数据类型元素
number_int4b_t =        AscendC::Std::get<0>(test2);
number_int8_t =         AscendC::Std::get<1>(test2);
number_uint8_t =        AscendC::Std::get<2>(test2);
number_int16_t =        AscendC::Std::get<3>(test2);
number_uint16_t =       AscendC::Std::get<4>(test2);
number_int32_t =        AscendC::Std::get<5>(test2);
number_uint32_t =       AscendC::Std::get<6>(test2);
number_uint64_t =       AscendC::Std::get<7>(test2);
number_int64_t =        AscendC::Std::get<8>(test2);
number_half =           AscendC::Std::get<9>(test2);
number_float =          AscendC::Std::get<10>(test2);
number_bfloat16_t =     AscendC::Std::get<11>(test2);
number_fp8_e8m0_t =     AscendC::Std::get<12>(test2);
number_bool =           AscendC::Std::get<13>(test2);

// get方法获取元素引用接续运算
AscendC::Std::get<1>(test2)+= 1 ;
AscendC::Std::get<2>(test2)+= 1 ;
AscendC::Std::get<3>(test2)+= 1 ;
AscendC::Std::get<4>(test2)+= 1 ;
AscendC::Std::get<5>(test2)+= 1 ;
AscendC::Std::get<6>(test2)+= 1 ;
AscendC::Std::get<7>(test2)+= 1 ;
AscendC::Std::get<8>(test2)+= 1 ;
AscendC::Std::get<10>(test2) += (float)1.0 ;

// 变量初始化列表聚合
AscendC::Std::tuple<AscendC::int4b_t, int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, uint64_t, int64_t, half, float, \
bfloat16_t, fp8_e8m0_t, bool> test3 = {
    number_int4b_t, number_int8_t, number_uint8_t, number_int16_t, number_uint16_t, number_int32_t, number_uint32_t, \
    number_uint64_t, number_int64_t, number_half, number_float, number_bfloat16_t, number_fp8_e8m0_t, number_bool, 
};

AscendC::PRINTF("tuple size is --> %d\n", AscendC::Std::tuple_size<decltype(test3)>::value);

uint32_t const_uint32_t = 0;
float const_float = 0.0;
bool const_bool = false;

// const常量类型
const AscendC::Std::tuple<uint32_t, float, bool> test4{11, 2.2, true};

// get方法获取const常量类型
const_uint32_t = AscendC::Std::get<0>(test4);
const_float = AscendC::Std::get<1>(test4);
const_bool = AscendC::Std::get<2>(test4);

AscendC::Std::tuple_element<0,decltype(test4)>::type first = 77;
AscendC::Std::tuple_element<1,decltype(test4)>::type second = 7.7;
AscendC::Std::tuple_element<2,decltype(test4)>::type third = false;

AscendC::PRINTF("The value of the test element is: %d, %f, %d\n", first, second, third);

AscendC::Std::tie(const_uint32_t, const_float, const_bool) = test4;

AscendC::PRINTF("The value of the test element is: %d, %f, %d\n", const_uint32_t, const_float, const_bool);

// const元素聚合
AscendC::Std::tuple<const uint32_t, const float, const bool> test5{33, 4.4, true};

// get方法获取const元素
const_uint32_t = AscendC::Std::get<0>(test5);
const_float = AscendC::Std::get<1>(test5);
const_bool = AscendC::Std::get<2>(test5);

const AscendC::Std::tuple<const uint32_t, const float, const bool> test6{33, 4.4, true};

const_uint32_t = AscendC::Std::get<0>(test6);
const_float = AscendC::Std::get<1>(test6);
const_bool = AscendC::Std::get<2>(test6);

// 默认构造初始化
AscendC::Std::tuple<\
uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, \
uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, \
uint32_t, uint32_t, uint32_t, uint8_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,\
uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, \
uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, \
uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, \
uint32_t, uint32_t, uint32_t, uint32_t> test7;

// 默认元素初始化聚合
auto test8 = AscendC::Std::make_tuple(\
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, \
    11, 12, 13, 14, 15, 16, 17, 18, 19, 20, \
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30, \
    31, 32, 33, 34, 35, 36, 37, 38, 39, 40, \
    41, 42, 43, 44, 45, 46, 47, 48, 49, 50, \
    51, 52, 53, 54, 55, 56, 57, 58, 59, 60, \
    61, 62, 63, 64);

// 运算符重载赋值
test7 = test8;

AscendC::PRINTF("tuple size is --> %d\n", AscendC::Std::tuple_size<decltype(test7)>::value);
AscendC::PRINTF("tuple size is --> %d\n", AscendC::Std::tuple_size<decltype(test8)>::value);

// make_tuple聚合初始化
AscendC::Std::tuple<uint32_t, float, bool> test9 = AscendC::Std::make_tuple(const_uint32_t, const_float, const_bool);

const_uint32_t = AscendC::Std::get<0>(test9);
const_float = AscendC::Std::get<1>(test9);
const_bool = AscendC::Std::get<2>(test9);

AscendC::PRINTF("tuple size is --> %d\n", AscendC::Std::tuple_size<decltype(test9)>::value);

using Element0Type = AscendC::Std::tuple_element<0, decltype(test9)>::type;
Element0Type element0 = 88;

using Element1Type = AscendC::Std::tuple_element<1, decltype(test9)>::type;
Element1Type element1 = 8.8;

using Element2Type = AscendC::Std::tuple_element<2, decltype(test9)>::type;
Element2Type element2 = true;

AscendC::PRINTF("The value of the test element is: %d, %f, %d\n", element0, element1, element2);

AscendC::Std::tie(const_uint32_t, const_float, const_bool) = test9;

AscendC::PRINTF("The value of the test element is: %d, %f, %d\n", const_uint32_t, const_float, const_bool);
```

```
// 执行结果：
(testMakeTensor) tuple size is --> 4
(test3Tensor) tuple size is --> 4
(test1) tuple size is --> 14
(test2) tuple size is --> 14
(test3) tuple size is --> 14
(test4 tuple_element) The value of the test element is: 77, 7.700000, 0
(test4 tie) The value of the test element is: 11, 2.200000, 1
(test7) tuple size is --> 64
(test8) tuple size is --> 64
(test9) tuple size is --> 3
(test9 tuple_element) The value of the test element is: 88, 8.800000, 1
(test9 tie) The value of the test element is: 33, 4.400000, 1
```
