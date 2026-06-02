# printf

**页面ID:** atlasascendc_api_07_00166  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_00166.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品            /             Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品            /             Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | x |
| Atlas 推理系列产品            AI Core | x |
| Atlas 推理系列产品            Vector Core | x |
| Atlas 训练系列产品 | x |

#### 功能说明

该接口提供AI CPU算子Kernel调试场景下的格式化输出功能，默认将输出内容解析并打印在屏幕上。使用时需要包含头文件aicpu_api.h。

#### 函数原型

```
void printf(const char* fmt, ...)
```

#### 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| fmt | 输入 | 格式控制字符串，包含两种类型的对象：普通字符和转换说明。                     - 普通字符将原样不动地打印输出。           - 转换说明并不直接输出而是用于控制printf中参数的转换和打印。每个转换说明都由一个百分号字符（%）开始，以转换说明结束，从而说明输出数据的类型 。                         支持的转换类型包括：                             - %d / %i：输出十进制数，支持打印的数据类型：bool/int8_t/int16_t/int32_t/int64_t                - %f：输出实数，支持打印的数据类型：float/half                - %x：输出十六进制整数，支持打印的数据类型：int8_t/int16_t/int32_t/int64_t/uint8_t/uint16_t/uint32_t/uint64_t                - %s：输出字符串                - %u：输出unsigned类型数据，支持打印的数据类型：bool/uint8_t/uint16_t/uint32_t/uint64_t                - %p：输出指针地址 |
| ... | 输入 | 附加参数，个数和类型可变的参数列表：根据不同的fmt字符串，函数可能需要一系列的附加参数，每个参数包含了一个要被插入的值，替换了fmt参数中指定的每个%标签。参数的个数应与%标签的个数相同。 |

#### 约束说明

- 该接口仅支持通过<<<...>>>调用，并在异构编译场景使用。
- 该接口不支持打印除换行符之外的其他转义字符。
- 该接口使用Dump功能，所有使用Dump功能的接口在每个核上Dump的数据总量不可超过1M。请开发者自行控制待打印的内容数据量，超出则不会打印。
- 使用该接口时，若采用bisheng命令行编译，开发者需要手动链接相关的静态库；而使用CMake编译时，框架会自动处理链接问题，无需开发者额外关注。具体编译命令如下：通过--cce-aicpu-laicpu_api为Device链接libaicpu_api.a，通过--cce-aicpu-L指定libaicpu_api.a的库路径。

```
$bisheng -O2 foo.aicpu --cce-aicpu-L${INSTALL_DIR}/lib64/device/lib64 --cce-aicpu-laicpu_api -I${INSTALL_DIR}/include/ascendc/aicpu_api -c -o foo.aicpu.o
```

${INSTALL_DIR}请替换为CANN软件安装后文件存储路径。以root用户安装为例，则安装后文件存储路径为：/usr/local/Ascend/cann。

#### 调用示例

在算子Kernel侧实现代码中需要输出日志信息的地方调用printf接口打印相关内容。样例如下：

```
#include "aicpu_api.h"

// 整型打印：
AscendC::printf("fmt string %d\n", 0x123);

// 浮点型打印：
float a = 3.14;
AscendC::printf("fmt string %f\n", a);

// 指针打印：
int b = 10;
int *c = &b;
AscendC::printf("TEST %p\n", c);
```

程序运行时打印效果如下：

```
fmt string 291
fmt string 3.140000
TEST 0xdfffd6fddd1c
```
