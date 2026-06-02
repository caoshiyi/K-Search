# ascendc_assert

**页面ID:** atlasascendc_api_07_00168  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_00168.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品AI Core | √ |
| Atlas 推理系列产品Vector Core | x |
| Atlas 训练系列产品 | x |

#### 功能说明

ascendc_assert提供了一种在CPU/NPU域实现断言功能的接口。当断言条件不满足时，系统会输出断言信息并格式化打印在屏幕上。

在算子Kernel侧实现代码中需要增加断言的地方使用ascendc_assert检查代码，并格式化输出一些调测信息。示例如下：

```
int assertFlag = 10;

ascendc_assert(assertFlag != 10);
ascendc_assert(assertFlag != 10, "The assertFlag value is 10.\n");
ascendc_assert(assertFlag != 10, "The assertFlag value is %d.\n", assertFlag);
```

> **注意:** 

ascendc_assert接口打印功能会对算子实际运行的性能带来一定影响（每一条ascendc_assert，系统会额外增加一条逻辑判断，具体性能影响取决于代码中ascendc_assert的使用数量），通常在调测阶段使用。开发者可以通过设置ASCENDC_DUMP=0来关闭打印功能。

NPU域ascendc_assert打印信息示例如下（DumpHead信息仅在使用自定义算子工程时才会打印）：

```
DumpHead: AIV-0, CoreType=AIV, block dim=8, total_block_num=8, block_remain_len=696, block_initial_space=1024, rsv=0, magic=5aa5bccd
[ASSERT] /home/.../add_custom.cpp:44: Assertion `assertFlag != 10' The assertFlag value is 10.
```

CPU域ascendc_assert打印信息示例如下：

```
DumpHead: AIV-0, CoreType=AIV, block dim=8, total_block_num=8, block_remain_len=696, block_initial_space=1024, rsv=0, magic=5aa5bccd
[ASSERT]/home/.../add_custom.cpp:44: Assertion `assertFlag != 10'
```

#### 函数原型

```
ascendc_assert(expr)
ascendc_assert(expr, __gm__ const char *fmt, Args&&... args)
```

#### 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| expr | 输入 | assert断言是否终止程序的条件。为true则程序继续执行，为false则终止程序。 |
| fmt | 输入 | 格式控制字符串，包含两种类型的对象：普通字符和转换说明。 - 普通字符将原样不动地打印输出。- 转换说明并不直接输出而是用于控制printf中参数的转换和打印。每个转换说明都由一个百分号字符（%）开始，以转换说明结束，从而说明输出数据的类型 。支持的转换类型包括：  - %d / %i：输出十进制数，支持打印的数据类型：bool/int8_t/int16_t/int32_t/int64_t  - %f：输出实数，支持打印的数据类型：float/half  - %x：输出十六进制整数，支持打印的数据类型：int8_t/int16_t/int32_t/int64_t/uint8_t/uint16_t/uint32_t/uint64_t  - %s：输出字符串  - %u：输出unsigned类型数据，支持打印的数据类型：bool/uint8_t/uint16_t/uint32_t/uint64_t  - %p：输出指针地址   **注意**：- 上文列出的数据类型是NPU域调试支持的数据类型，CPU域调试时，支持的数据类型和C/C++规范保持一致。- 在转换类型为%x，即输出十六进制整数时，NPU域上的输出为64位，CPU域上的输出为32位。 |
| args | 输入 | 附加参数，个数和类型可变的参数列表：根据不同的fmt字符串，函数可能需要一系列的附加参数，每个参数包含了一个要被插入的值，替换了fmt参数中指定的每个%标签。参数的个数应与%标签的个数相同。 |

#### 约束说明

- 该接口不支持算子入图场景。
- 该接口不支持simulator仿真模式。
- 该接口不需要使用AscendC命名空间。

- 该接口不支持打印除换行符之外的其他转义字符。
- 单次调用本接口打印的数据总量不可超过1MB（还包括少量框架需要的头尾信息，通常可忽略）。使用时应注意，如果超出这个限制，则数据不会被打印。
- 使用自定义算子工程时，存在以下限制：

  - 该接口使用Dump功能，一个算子所有使用Dump功能的接口在每个核上Dump的数据总量不可超过1M。请开发者自行控制待打印的内容数据量，超出则不会打印。
  - 该接口使用空间每个核上不可超过1k。请开发者自行控制待打印的内容数据量，超出则不会打印。

#### 调用示例

- 接口调用示例

```
int assertFlag = 10;

// 断言条件
ascendc_assert(assertFlag == 10);

// 打印消息
ascendc_assert(assertFlag == 10, "The assertFlag value is 10.\n");

// 格式化打印
ascendc_assert(assertFlag == 10, "The assertFlag value is %d.\n", assertFlag);
ascendc_assert(assertFlag != 10, "The assertFlag value is %d.\n", assertFlag);
```

程序运行时会触发assert，打印效果如下：

```
[ASSERT] /home/.../add_custom.cpp:44: Assertion `assertFlag != 10' The assertFlag value is 10.
```

- 附：自定义算子工程和Kernel直调工程中关闭关闭ASCENDC_DUMP开关的示例。

  - 自定义算子工程修改算子工程op_kernel目录下的CMakeLists.txt文件，首行增加编译选项-DASCENDC_DUMP=0，关闭ASCENDC_DUMP开关，示例如下：

```
// 关闭所有算子的打印功能
add_ops_compile_options(ALL OPTIONS -DASCENDC_DUMP=0)
```

  - Kernel直调工程

    - 修改cmake目录下的npu_lib.cmake文件，在ascendc_compile_definitions命令中增加ASCENDC_DUMP=0宏定义来关闭NPU侧ASCENDC_DUMP开关。示例如下：

```
// 关闭所有算子的打印功能
ascendc_compile_definitions(ascendc_kernels_${RUN_MODE} PRIVATE
    ASCENDC_DUMP=0
)
```

    - 修改cmake目录下的cpu_lib.cmake文件，在target_compile_definitions命令中增加ASCENDC_DUMP=0宏定义来关闭CPU侧ASCENDC_DUMP开关。示例如下：

```
target_compile_definitions(ascendc_kernels_${RUN_MODE} PRIVATE
    ASCENDC_DUMP=0
)
```
