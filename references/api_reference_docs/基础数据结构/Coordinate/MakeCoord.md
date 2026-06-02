# MakeCoord

**页面ID:** atlasascendc_api_07_00145  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_00145.html

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

将传入的数据打包成Coord数据结构。

#### 函数原型

```
template <typename... Ts>
__aicore__ inline constexpr Coord<Ts...> MakeCoord(Ts const&... t)
```

#### 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| t | 输入 | 表示输入数据类型的形参包，具体使用方法和约束说明同Std::tuple。 输入的数据类型支持size_t和Std::Int。 |

#### 返回值说明

Coord结构类型（Std::tuple类型的别名）。

#### 约束说明

无

#### 调用示例

```
auto blockCoordM = Std::Int<11>{};
auto blockCoordN = Std::Int<12>{};
auto coord = AscendC::MakeCoord(blockCoordM, blockCoordN);
```
