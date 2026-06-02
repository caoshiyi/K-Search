# TPosition

**页面ID:** atlasascendc_api_07_0174  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0174.html

---

Ascend C管理不同层级的物理内存时，用一种抽象的逻辑位置（TPosition）来表达各级别的存储，代替了片上物理存储的概念，达到隐藏硬件架构的目的。主要的TPosition类型包括：VECIN、VECOUT、VECCALC、A1、A2、B1、B2、C1、C2、CO1、CO2，其中VECIN、VECCALC、VECOUT主要用于矢量编程，A1、A2、B1、B2、C1、C2、CO1、CO2用于矩阵编程。您可以参考编程范式了解TPosition的基础概念，通过表1了解TPosition和物理存储的映射关系。

TPosition定义如下：

```
enum class TPosition : uint8_t {
    GM,
    A1,
    A2,
    B1,
    B2,
    C1,
    C2,
    CO1,
    CO2,
    VECIN,
    VECOUT,
    VECCALC,
    LCM = VECCALC,
    SPM,
    SHM = SPM,
    TSCM,
    C2PIPE2GM,
    C2PIPE2LOCAL,
    MAX,
};
```

TPosition枚举值的具体定义如下：

**表1 **TPosition枚举值含义说明

| 枚举值 | 具体含义 |
| --- | --- |
| GM | Global Memory，对应AI Core的外部存储。 |
| VECIN | 用于矢量计算，搬入数据的存放位置，在数据搬入Vector计算单元时使用此位置。 |
| VECOUT | 用于矢量计算，搬出数据的存放位置，在将Vector计算单元结果搬出时使用此位置。 |
| VECCALC | 用于矢量计算/矩阵计算，在计算需要临时变量时使用此位置。 |
| A1 | 用于矩阵计算，存放整块A矩阵，可类比CPU多级缓存中的二级缓存。 |
| B1 | 用于矩阵计算，存放整块B矩阵，可类比CPU多级缓存中的二级缓存。 |
| C1 | 用于矩阵计算，存放整块Bias矩阵，可类比CPU多级缓存中的二级缓存。 |
| A2 | 用于矩阵计算，存放切分后的小块A矩阵，可类比CPU多级缓存中的一级缓存。 |
| B2 | 用于矩阵计算，存放切分后的小块B矩阵，可类比CPU多级缓存中的一级缓存。 |
| C2 | 用于矩阵计算，存放切分后的小块Bias矩阵，可类比CPU多级缓存中的一级缓存。 |
| CO1 | 用于矩阵计算，存放小块结果C矩阵，可理解为Cube Out。 |
| CO2 | 用于矩阵计算，存放整块结果C矩阵，可理解为Cube Out。 |
| LCM | Local Cache Memory，代表临时共享的Unified Buffer空间，VECCALC的别名，与VECCALC实现同样的功能。 |
| SPM | 当Unified Buffer内存有溢出风险时，用于Unified Buffer的数据暂存。 |
| SHM | SPM的别名。 |
| TSCM | Temp Swap Cache Memory，用于临时把数据交换到额外空间，进行Matmul运算。 |
| C2PIPE2GM | 用于存放FIXPIPE量化参数。 |
| C2PIPE2LOCAL | 预留参数。为后续的功能做保留，开发者暂时无需关注。 |
