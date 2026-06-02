# Kernel编译时报错“error: out of jump/jumpc imm range”

**页面ID:** atlas_ascendc_10_0109  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_10_0109.html

---

#### 现象描述

使用工程化算子开发方式，基于自定义算子工程进行算子开发。编译算子时失败，报如下错误：

```
[ERROR] [ascendxxxx] PowerCustom_88a695f03edfbc0af76b9eaae9e4556c error: out of jump/jumpc imm range
```

#### 问题根因

该编译错误的原因是算子kernel代码过大，导致在编译时跳转指令跳转的偏移值超过了限定的大小(int16_t的数据范围)，可通过添加编译选项“-mllvm -cce-aicore-jump-expand=true”通过间接跳转的方式来避免该问题，让编译器能够正常编译。

#### 处理步骤

1. 在kernel侧的CMakeLists中通过add_ops_compile_options针对报错算子添加编译选项“-mllvm -cce-aicore-jump-expand=true”，示例如下：

```
add_ops_compile_options(PowerCustom OPTIONS -mllvm -cce-aicore-jump-expand=true)
```

add_ops_compile_options的具体使用方法请参考支持自定义编译选项。

2. 重新编译该算子。正常编译无报错。
