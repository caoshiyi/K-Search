# basic_knowledge 文档索引

## 入门教程

### 快速入门

- [Add自定义算子开发](入门教程/快速入门/Add自定义算子开发.md)
- [HelloWorld](入门教程/快速入门/HelloWorld.md)

- [什么是Ascend C](入门教程/什么是Ascend_C.md)
- [环境准备](入门教程/环境准备.md)

## 算子实践参考

### SIMD算子实现

#### 矢量编程

##### 多核&Tiling切分

- [多核Tiling](算子实践参考/SIMD算子实现/矢量编程/多核&Tiling切分/多核Tiling.md)
- [尾块Tiling](算子实践参考/SIMD算子实现/矢量编程/多核&Tiling切分/尾块Tiling.md)
- [尾核&尾块](算子实践参考/SIMD算子实现/矢量编程/多核&Tiling切分/尾核&尾块.md)
- [尾核Tiling](算子实践参考/SIMD算子实现/矢量编程/多核&Tiling切分/尾核Tiling.md)
- [概述](算子实践参考/SIMD算子实现/矢量编程/多核&Tiling切分/概述.md)

- [Broadcast场景](算子实践参考/SIMD算子实现/矢量编程/Broadcast场景.md)
- [DoubleBuffer场景](算子实践参考/SIMD算子实现/矢量编程/DoubleBuffer场景.md)
- [TBuf的使用](算子实践参考/SIMD算子实现/矢量编程/TBuf的使用.md)
- [基础矢量算子](算子实践参考/SIMD算子实现/矢量编程/基础矢量算子.md)
- [非对齐场景](算子实践参考/SIMD算子实现/矢量编程/非对齐场景.md)

#### 矩阵编程（基础API）

- [分离模式](算子实践参考/SIMD算子实现/矩阵编程（基础API）/分离模式.md)
- [耦合模式](算子实践参考/SIMD算子实现/矩阵编程（基础API）/耦合模式.md)

#### 矩阵编程（高阶API）

##### 特性场景

- [4:2稀疏矩阵乘](算子实践参考/SIMD算子实现/矩阵编程（高阶API）/特性场景/4_2稀疏矩阵乘.md)
- [AIC和AIV独立运行机制](算子实践参考/SIMD算子实现/矩阵编程（高阶API）/特性场景/AIC和AIV独立运行机制.md)
- [Batch Matmul基础功能](算子实践参考/SIMD算子实现/矩阵编程（高阶API）/特性场景/Batch_Matmul基础功能.md)
- [Batch Matmul复用Bias矩阵](算子实践参考/SIMD算子实现/矩阵编程（高阶API）/特性场景/Batch_Matmul复用Bias矩阵.md)
- [Matmul特性介绍](算子实践参考/SIMD算子实现/矩阵编程（高阶API）/特性场景/Matmul特性介绍.md)
- [TSCM输入的矩阵乘](算子实践参考/SIMD算子实现/矩阵编程（高阶API）/特性场景/TSCM输入的矩阵乘.md)
- [单次矩阵乘局部输出](算子实践参考/SIMD算子实现/矩阵编程（高阶API）/特性场景/单次矩阵乘局部输出.md)
- [多核对齐切分](算子实践参考/SIMD算子实现/矩阵编程（高阶API）/特性场景/多核对齐切分.md)
- [多核非对齐切分](算子实践参考/SIMD算子实现/矩阵编程（高阶API）/特性场景/多核非对齐切分.md)
- [异步场景处理](算子实践参考/SIMD算子实现/矩阵编程（高阶API）/特性场景/异步场景处理.md)
- [矩阵乘输出的Channel拆分](算子实践参考/SIMD算子实现/矩阵编程（高阶API）/特性场景/矩阵乘输出的Channel拆分.md)
- [矩阵乘输出的N方向对齐](算子实践参考/SIMD算子实现/矩阵编程（高阶API）/特性场景/矩阵乘输出的N方向对齐.md)
- [矩阵乘输出的量化/反量化](算子实践参考/SIMD算子实现/矩阵编程（高阶API）/特性场景/矩阵乘输出的量化_反量化.md)
- [矩阵向量乘](算子实践参考/SIMD算子实现/矩阵编程（高阶API）/特性场景/矩阵向量乘.md)

- [基础知识](算子实践参考/SIMD算子实现/矩阵编程（高阶API）/基础知识.md)
- [算子实现](算子实践参考/SIMD算子实现/矩阵编程（高阶API）/算子实现.md)

#### 融合算子编程

##### CV融合

- [基础知识](算子实践参考/SIMD算子实现/融合算子编程/CV融合/基础知识.md)
- [算子实现](算子实践参考/SIMD算子实现/融合算子编程/CV融合/算子实现.md)

##### 通算融合

- [基础知识](算子实践参考/SIMD算子实现/融合算子编程/通算融合/基础知识.md)
- [特性场景](算子实践参考/SIMD算子实现/融合算子编程/通算融合/特性场景.md)
- [算子实现](算子实践参考/SIMD算子实现/融合算子编程/通算融合/算子实现.md)

- [概述](算子实践参考/SIMD算子实现/概述.md)

### SIMD算子性能优化

#### Tiling策略

- [核间负载均衡](算子实践参考/SIMD算子性能优化/Tiling策略/核间负载均衡.md)

#### 内存访问

- [GM地址尽量512B对齐](算子实践参考/SIMD算子性能优化/内存访问/GM地址尽量512B对齐.md)
- [L2 Cache切分](算子实践参考/SIMD算子性能优化/内存访问/L2_Cache切分.md)
- [尽量一次搬运较大的数据块](算子实践参考/SIMD算子性能优化/内存访问/尽量一次搬运较大的数据块.md)
- [算子与高阶API共享临时Buffer](算子实践参考/SIMD算子性能优化/内存访问/算子与高阶API共享临时Buffer.md)
- [纯搬运类算子VECIN和VECOUT建议复用](算子实践参考/SIMD算子性能优化/内存访问/纯搬运类算子VECIN和VECOUT建议复用.md)
- [设置合理的L2 CacheMode](算子实践参考/SIMD算子性能优化/内存访问/设置合理的L2_CacheMode.md)
- [通过缩减Tensor ShapeInfo维度，优化栈空间](算子实践参考/SIMD算子性能优化/内存访问/通过缩减Tensor_ShapeInfo维度，优化栈空间.md)
- [避免Unified Buffer的bank冲突](算子实践参考/SIMD算子性能优化/内存访问/避免Unified_Buffer的bank冲突.md)
- [避免同地址访问](算子实践参考/SIMD算子性能优化/内存访问/避免同地址访问.md)
- [高效的使用搬运API](算子实践参考/SIMD算子性能优化/内存访问/高效的使用搬运API.md)

#### 头尾开销优化

- [设置合适的核数和算子Kernel类型](算子实践参考/SIMD算子性能优化/头尾开销优化/设置合适的核数和算子Kernel类型.md)
- [避免TPipe在对象内创建和初始化](算子实践参考/SIMD算子性能优化/头尾开销优化/避免TPipe在对象内创建和初始化.md)
- [限制TilingData结构大小](算子实践参考/SIMD算子性能优化/头尾开销优化/限制TilingData结构大小.md)

#### 流水编排

- [使能DoubleBuffer](算子实践参考/SIMD算子性能优化/流水编排/使能DoubleBuffer.md)
- [使能Iterate或IterateAll异步接口避免AIC/AIV同步依赖](算子实践参考/SIMD算子性能优化/流水编排/使能Iterate或IterateAll异步接口避免AIC_AIV同步依赖.md)

#### 矢量计算

- [Vector算子灵活运用Counter模式](算子实践参考/SIMD算子性能优化/矢量计算/Vector算子灵活运用Counter模式.md)
- [选择低延迟指令，优化归约操作性能](算子实践参考/SIMD算子性能优化/矢量计算/选择低延迟指令，优化归约操作性能.md)
- [通过Unified Buffer融合实现连续vector计算](算子实践参考/SIMD算子性能优化/矢量计算/通过Unified_Buffer融合实现连续vector计算.md)

#### 矩阵计算

- [Matmul使能AtomicAdd选项](算子实践参考/SIMD算子性能优化/矩阵计算/Matmul使能AtomicAdd选项.md)
- [较小矩阵长驻L1 Buffer，仅分次搬运较大矩阵](算子实践参考/SIMD算子性能优化/矩阵计算/较小矩阵长驻L1_Buffer，仅分次搬运较大矩阵.md)
- [通过BT Buffer实现高效的bias计算](算子实践参考/SIMD算子性能优化/矩阵计算/通过BT_Buffer实现高效的bias计算.md)
- [通过FP Buffer存放量化参数实现高效随路量化](算子实践参考/SIMD算子性能优化/矩阵计算/通过FP_Buffer存放量化参数实现高效随路量化.md)
- [通过L0C Buffer数据暂存实现高效的矩阵乘结果累加](算子实践参考/SIMD算子性能优化/矩阵计算/通过L0C_Buffer数据暂存实现高效的矩阵乘结果累加.md)

- [优化建议总览表](算子实践参考/SIMD算子性能优化/优化建议总览表.md)

### 优秀实践

#### Matmul性能调优案例

- [AIV核上的ND2NZ格式转换](算子实践参考/优秀实践/Matmul性能调优案例/AIV核上的ND2NZ格式转换.md)
- [Matmul性能优化策略总览](算子实践参考/优秀实践/Matmul性能调优案例/Matmul性能优化策略总览.md)
- [Matmul算子优化Tiling策略](算子实践参考/优秀实践/Matmul性能调优案例/Matmul算子优化Tiling策略.md)
- [Matmul高阶API使能IBShare模板共享A和B矩阵数据](算子实践参考/优秀实践/Matmul性能调优案例/Matmul高阶API使能IBShare模板共享A和B矩阵数据.md)
- [Matmul高阶API使能IBShare模板共享B矩阵数据](算子实践参考/优秀实践/Matmul性能调优案例/Matmul高阶API使能IBShare模板共享B矩阵数据.md)
- [Matmul高阶API使能L2 Cache切分](算子实践参考/优秀实践/Matmul性能调优案例/Matmul高阶API使能L2_Cache切分.md)
- [Matmul高阶API使能MDL模板](算子实践参考/优秀实践/Matmul性能调优案例/Matmul高阶API使能MDL模板.md)
- [Matmul高阶API使能MTE2 Preload](算子实践参考/优秀实践/Matmul性能调优案例/Matmul高阶API使能MTE2_Preload.md)
- [Matmul高阶API使能NBuffer33模板](算子实践参考/优秀实践/Matmul性能调优案例/Matmul高阶API使能NBuffer33模板.md)
- [Matmul高阶API使能Tiling全量常量化](算子实践参考/优秀实践/Matmul性能调优案例/Matmul高阶API使能Tiling全量常量化.md)
- [Matmul高阶API使能UnitFlag](算子实践参考/优秀实践/Matmul性能调优案例/Matmul高阶API使能UnitFlag.md)
- [Matmul高阶API使能多核K轴错峰访问内存](算子实践参考/优秀实践/Matmul性能调优案例/Matmul高阶API使能多核K轴错峰访问内存.md)
- [Matmul高阶API使能多核切K](算子实践参考/优秀实践/Matmul性能调优案例/Matmul高阶API使能多核切K.md)
- [Matmul高阶API使能纯Cube模式](算子实践参考/优秀实践/Matmul性能调优案例/Matmul高阶API使能纯Cube模式.md)

- [FlashAttention算子性能调优案例](算子实践参考/优秀实践/FlashAttention算子性能调优案例.md)
- [GroupedMatmul算子性能调优案例](算子实践参考/优秀实践/GroupedMatmul算子性能调优案例.md)
- [MC²算子性能调优案例](算子实践参考/优秀实践/MC²算子性能调优案例.md)

### 功能调试

- [精度正常](算子实践参考/功能调试/精度正常.md)

### 性能分析

- [分析性能数据](算子实践参考/性能分析/分析性能数据.md)
- [获取性能数据](算子实践参考/性能分析/获取性能数据.md)

- [异构计算](算子实践参考/异构计算.md)
- [本文档组织结构](算子实践参考/本文档组织结构.md)

## 编程指南

### C++类库API

#### 基础API

##### 常用操作速查指导

- [如何使用归约计算API](编程指南/C++类库API/基础API/常用操作速查指导/如何使用归约计算API.md)
- [如何使用掩码操作API](编程指南/C++类库API/基础API/常用操作速查指导/如何使用掩码操作API.md)

##### 接口分类说明

- [连续计算API](编程指南/C++类库API/基础API/接口分类说明/连续计算API.md)
- [高维切分API](编程指南/C++类库API/基础API/接口分类说明/高维切分API.md)

- [概述](编程指南/C++类库API/基础API/概述.md)

#### 高阶API

##### 常用操作速查指导

- [如何使用Kernel侧临时空间](编程指南/C++类库API/高阶API/常用操作速查指导/如何使用Kernel侧临时空间.md)
- [如何使用Tiling依赖的头文件](编程指南/C++类库API/高阶API/常用操作速查指导/如何使用Tiling依赖的头文件.md)

- [编程接口概述](编程指南/C++类库API/编程接口概述.md)

### 概念原理和术语

#### 内存访问原理

- [Scalar读写数据](编程指南/概念原理和术语/内存访问原理/Scalar读写数据.md)

#### 性能优化技术原理

- [DoubleBuffer](编程指南/概念原理和术语/性能优化技术原理/DoubleBuffer.md)

#### 神经网络和算子

- [数据排布格式](编程指南/概念原理和术语/神经网络和算子/数据排布格式.md)
- [算子基本概念](编程指南/概念原理和术语/神经网络和算子/算子基本概念.md)

- [术语表](编程指南/概念原理和术语/术语表.md)
- [编程模型设计原理](编程指南/概念原理和术语/编程模型设计原理.md)

### 硬件实现

#### 架构规格

- [NPU架构版本200x](编程指南/硬件实现/架构规格/NPU架构版本200x.md)
- [NPU架构版本220x](编程指南/硬件实现/架构规格/NPU架构版本220x.md)
- [NPU架构版本300x](编程指南/硬件实现/架构规格/NPU架构版本300x.md)

- [基本架构](编程指南/硬件实现/基本架构.md)
- [硬件约束](编程指南/硬件实现/硬件约束.md)

### 编程模型

#### 编程范式

##### AI Core SIMD编程

###### 基于TPipe和TQue编程

- [典型算子的编程范式](编程指南/编程模型/编程范式/AI_Core_SIMD编程/基于TPipe和TQue编程/典型算子的编程范式.md)

- [静态Tensor编程](编程指南/编程模型/编程范式/AI_Core_SIMD编程/静态Tensor编程.md)

- [AI CPU编程](编程指南/编程模型/编程范式/AI_CPU编程.md)

- [异构并行编程模型](编程指南/编程模型/异构并行编程模型.md)
- [抽象硬件架构](编程指南/编程模型/抽象硬件架构.md)
- [核函数](编程指南/编程模型/核函数.md)

### 编译与运行

#### AI Core算子编译

- [RTC](编程指南/编译与运行/AI_Core算子编译/RTC.md)
- [常用的编译选项](编程指南/编译与运行/AI_Core算子编译/常用的编译选项.md)
- [算子编译简介](编程指南/编译与运行/AI_Core算子编译/算子编译简介.md)
- [通过CMake编译](编程指南/编译与运行/AI_Core算子编译/通过CMake编译.md)
- [通过bisheng命令行编译](编程指南/编译与运行/AI_Core算子编译/通过bisheng命令行编译.md)

- [AI CPU算子编译](编程指南/编译与运行/AI_CPU算子编译.md)
- [算子运行](编程指南/编译与运行/算子运行.md)

### 语言扩展层

- [SIMD BuiltIn关键字和API](编程指南/语言扩展层/SIMD_BuiltIn关键字和API.md)

### 调试调优

#### 功能调试

- [CPU域孪生调试](编程指南/调试调优/功能调试/CPU域孪生调试.md)
- [NPU域上板调试](编程指南/调试调优/功能调试/NPU域上板调试.md)

- [性能调优](编程指南/调试调优/性能调优.md)
- [概述](编程指南/调试调优/概述.md)

### 附录

#### AI框架算子适配

##### ONNX框架

- [调用样例](编程指南/附录/AI框架算子适配/ONNX框架/调用样例.md)
- [适配插件开发](编程指南/附录/AI框架算子适配/ONNX框架/适配插件开发.md)

- [PyTorch框架](编程指南/附录/AI框架算子适配/PyTorch框架.md)
- [TensorFlow框架](编程指南/附录/AI框架算子适配/TensorFlow框架.md)

#### FAQ

- [Kernel编译时报错“error: out of jump/jumpc imm range”](编程指南/附录/FAQ/Kernel编译时报错“error_out_of_jump_jumpc_imm_range”.md)
- [kernel侧获取Tiling信息不正确](编程指南/附录/FAQ/kernel侧获取Tiling信息不正确.md)
- [使用跨版本的自定义算子包时，含有Matmul高阶API的算子存在编译或执行报错](编程指南/附录/FAQ/使用跨版本的自定义算子包时，含有Matmul高阶API的算子存在编译或执行报错.md)
- [含有Matmul高阶API的算子精度问题](编程指南/附录/FAQ/含有Matmul高阶API的算子精度问题.md)
- [核函数运行验证时算子存在精度问题](编程指南/附录/FAQ/核函数运行验证时算子存在精度问题.md)
- [算子包部署时出现权限不足报错](编程指南/附录/FAQ/算子包部署时出现权限不足报错.md)
- [算子工程编译时出现文件名过长报错](编程指南/附录/FAQ/算子工程编译时出现文件名过长报错.md)
- [调用算子时出现无法打开config.ini的报错](编程指南/附录/FAQ/调用算子时出现无法打开config.ini的报错.md)
- [运行验证时AllocTensor/FreeTensor失败](编程指南/附录/FAQ/运行验证时AllocTensor_FreeTensor失败.md)

#### 工程化算子开发

##### Host侧Tiling实现

- [Tiling模板编程](编程指南/附录/工程化算子开发/Host侧Tiling实现/Tiling模板编程.md)
- [使用标准C++语法定义Tiling结构体](编程指南/附录/工程化算子开发/Host侧Tiling实现/使用标准C++语法定义Tiling结构体.md)
- [使用高阶API时配套的Tiling实现](编程指南/附录/工程化算子开发/Host侧Tiling实现/使用高阶API时配套的Tiling实现.md)
- [基本流程](编程指南/附录/工程化算子开发/Host侧Tiling实现/基本流程.md)
- [通过TilingData传递属性信息](编程指南/附录/工程化算子开发/Host侧Tiling实现/通过TilingData传递属性信息.md)

##### 算子包编译

- [算子包部署](编程指南/附录/工程化算子开发/算子包编译/算子包部署.md)
- [算子工程编译](编程指南/附录/工程化算子开发/算子包编译/算子工程编译.md)

- [Kernel侧算子实现](编程指南/附录/工程化算子开发/Kernel侧算子实现.md)
- [创建算子工程](编程指南/附录/工程化算子开发/创建算子工程.md)
- [单算子API调用](编程指南/附录/工程化算子开发/单算子API调用.md)
- [概述](编程指南/附录/工程化算子开发/概述.md)
- [算子动态库和静态库编译](编程指南/附录/工程化算子开发/算子动态库和静态库编译.md)
- [算子原型定义](编程指南/附录/工程化算子开发/算子原型定义.md)
- [算子工程编译拓展](编程指南/附录/工程化算子开发/算子工程编译拓展.md)

#### 常用操作

- [如何使用Tensor原地操作提升算子性能](编程指南/附录/常用操作/如何使用Tensor原地操作提升算子性能.md)
- [如何使用workspace](编程指南/附录/常用操作/如何使用workspace.md)
- [如何在矢量编程时使能Vector Core](编程指南/附录/常用操作/如何在矢量编程时使能Vector_Core.md)
- [如何开发动态输入算子](编程指南/附录/常用操作/如何开发动态输入算子.md)
- [如何进行Tiling调测](编程指南/附录/常用操作/如何进行Tiling调测.md)

#### 算子入图（GE图）开发

- [SuperKernel开发](编程指南/附录/算子入图（GE图）开发/SuperKernel开发.md)
- [使能Tiling下沉](编程指南/附录/算子入图（GE图）开发/使能Tiling下沉.md)
- [图编译和图执行](编程指南/附录/算子入图（GE图）开发/图编译和图执行.md)
- [基本开发流程](编程指南/附录/算子入图（GE图）开发/基本开发流程.md)
- [概述](编程指南/附录/算子入图（GE图）开发/概述.md)

- [C++标准支持](编程指南/附录/C++标准支持.md)
- [msobjdump工具](编程指南/附录/msobjdump工具.md)
- [show_kernel_debug_data工具](编程指南/附录/show_kernel_debug_data工具.md)
- [基于样例工程完成Kernel直调](编程指南/附录/基于样例工程完成Kernel直调.md)
- [简易自定义算子工程](编程指南/附录/简易自定义算子工程.md)

