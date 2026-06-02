# is_layout

**页面ID:** atlasascendc_api_07_00085  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_00085.html

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

判断输入的数据结构是否为Layout数据结构，可通过检查其成员常量value的值来判断。当value为true时，表示输入的数据结构是Layout类型；反之则为非Layout类型。

#### 函数原型

```
template <typename T> struct is_layout
```

#### 参数说明

**表1 **模板参数说明

| 参数名 | 描述 |
| --- | --- |
| T | 根据输入的数据类型，判断是否为Layout数据结构。 |

#### 约束说明

无

#### 调用示例

```
// 初始化Layout数据结构并判断其类型
AscendC::Shape<int,int,int> shape = AscendC::MakeShape(10, 20, 30);
AscendC::Stride<int,int,int> stride = AscendC::MakeStride(1, 100, 200);

auto layoutMake = AscendC::MakeLayout(shape, stride);
AscendC::Layout<AscendC::Shape<int, int, int>, AscendC::Stride<int, int, int>> layoutInit(shape, stride);

bool value = AscendC::is_layout<decltype(shape)>::value; //value = false
value = AscendC::is_layout<decltype(stride)>::value; //value = false

value = AscendC::is_layout<decltype(layoutMake)>::value;//value = true
value = AscendC::is_layout<decltype(layoutInit)>::value;//value = true
```
