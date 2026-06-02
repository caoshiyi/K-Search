# RoundMode<a name="ZH-CN_TOPIC_0000002523344704"></a>

控制舍入模式，可输入值：

RoundMode::CAST\_RINT：返回最接近参数的整数，如果有2个整数同样接近，则会返回偶数的那个；

RoundMode::CAST\_ROUND：round模式， 四舍五入求整；

RoundMode::CAST\_FLOOR：floor模式，向下取整；

RoundMode::CAST\_CEIL：ceil模式， 向上取整；

RoundMode::CAST\_TRUNC：truncation模式， 截断取整；

RoundMode::CAST\_ODD：向奇数的方向舍入，既当小数点后数值不为0时，如果整数位是偶数，则进位；

RoundMode::CAST\_HYBRID：随机舍入，算子中目前特指输出结果是hif8数据时，会用到的一种随机舍入。

```
enum class RoundMode {
    CAST_NONE = 0,  // 在转换有精度损失时表示CAST_RINT模式，不涉及精度损失时表示不舍入
    CAST_RINT,      // rint，四舍六入五成双舍入
    CAST_FLOOR,     // floor，向负无穷舍入
    CAST_CEIL,      // ceil，向正无穷舍入
    CAST_ROUND,     // round，四舍五入舍入
    CAST_TRUNC,     // trunc，向零舍入
    CAST_ODD,       // Von Neumann rounding，最近邻奇数舍入
    CAST_HYBRID,    // hybrid，目前特指输出结果是hif8数据时，会用到的一种随机舍入 
};
```

