# is_tensorTrait

**页面ID:** atlasascendc_api_07_00125  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_00125.html

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

判断输入的数据结构是否为TensorTrait数据结构，可通过检查其成员常量value的值来判断。当value为true时，表示输入的数据结构是TensorTrait类型；反之则为非TensorTrait类型。

#### 函数原型

```
template <typename T> struct is_tensorTrait
```

#### 参数说明

| 参数名 | 描述 |
| --- | --- |
| T | 根据输入的数据类型，判断是否为TensorTrait数据结构。 |

#### 约束说明

无

#### 调用示例

```
// 以下示例为基于googletest的UT示例

AscendC::Shape<int,int,int> shape = AscendC::MakeShape(10, 20, 30);
AscendC::Stride<int,int,int> stride = AscendC::MakeStride(1, 100, 200);

auto layoutMake = AscendC::MakeLayout(shape, stride);    
auto tensorTraitMake = AscendC::MakeTensorTrait<float, AscendC::TPosition::VECIN>(layoutMake);

EXPECT_EQ(AscendC::is_tensorTrait<decltype(shape)>::value, false);
EXPECT_EQ(AscendC::is_tensorTrait<decltype(stride)>::value, false);
EXPECT_EQ(AscendC::is_tensorTrait<decltype(layoutMake)>::value, false);
EXPECT_EQ(AscendC::is_tensorTrait<decltype(tensorTraitMake)>::value, true);
```
