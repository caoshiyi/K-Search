# 运算符重载

**页面ID:** atlasascendc_api_07_00147  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_00147.html

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

重载运算符()，内部实现为调用Crd2Idx接口，将多维坐标（Coordinate）通过布局（Layout）转换为内存位置索引（Index）。

#### 函数原型

```
template <typename CoordType>
__aicore__ inline constexpr auto operator()(const CoordType& coord) const {}
```

#### 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| coord | 输入 | Coord结构类型，用于表示张量在不同维度上的坐标值。 |

#### 返回值说明

返回根据Coordinate信息转换之后的索引值。

#### 约束说明

无

#### 调用示例

```
...
auto layout = AscendC::MakeLayout(shape, stride);
auto index = layout(coord);
```
