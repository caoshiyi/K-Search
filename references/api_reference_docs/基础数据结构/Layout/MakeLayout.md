# MakeLayout

**页面ID:** atlasascendc_api_07_00084  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_00084.html

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

将传入的Shape和Stride数据打包成Layout数据结构。

#### 函数原型

```
template <typename ShapeType, typename StrideType>
__aicore__ inline constexpr auto MakeLayout(const ShapeType& shape, const StrideType& stride)
```

#### 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| shape | 输入 | Std::tuple结构类型，用于定义数据的逻辑形状，例如二维矩阵的行数和列数或多维张量的各维度大小。 |
| stride | 输入 | Std::tuple结构类型，用于定义各维度在内存中的步长，即同维度相邻元素在内存中的间隔，间隔的单位为元素，与Shape的维度信息一一对应。 |

#### 返回值说明

返回Layout对象。

#### 约束说明

构造Layout对象时传入的Shape和Stride结构，需是Std::tuple结构类型，且满足Std::tuple结构类型的使用约束。

#### 调用示例

```
// 初始化Layout数据结构，获取对应数值
AscendC::Shape<int,int,int> shape = AscendC::MakeShape(10, 20, 30);
AscendC::Stride<int,int,int> stride = AscendC::MakeStride(1, 100, 200);

auto layoutMake = AscendC::MakeLayout(shape, stride);
AscendC::Layout<AscendC::Shape<int, int, int>, AscendC::Stride<int, int, int>> layoutInit(shape, stride);

int value = AscendC::Std::get<0>(layoutMake.GetShape()); // value = 10
value = AscendC::Std::get<1>(layoutMake.GetShape()); // value = 20
value = AscendC::Std::get<2>(layoutMake.GetShape()); // value = 30

value = AscendC::Std::get<0>(layoutInit.GetStride()); // value = 1
value = AscendC::Std::get<1>(layoutInit.GetStride()); // value = 100
value = AscendC::Std::get<2>(layoutInit.GetStride()); // value = 200
```
