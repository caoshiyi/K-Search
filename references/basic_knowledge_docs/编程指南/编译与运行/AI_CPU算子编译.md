# AI CPU算子编译

**页面ID:** atlas_ascendc_10_00050  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_10_00050.html

---

#### 通过bisheng命令行编译

下文基于一个Hello World打印样例来讲解如何通过bisheng命令行编译AI CPU算子。

hello_world.aicpu文件内容如下：

```
#include "aicpu_api.h"

__global__ __aicpu__ uint32_t hello_world(void *args)
{
    AscendC::printf("Hello World!!!\n");
    return 0;
}
```

Host侧使用内核调用符<<<...>>>进行AI CPU算子的调用， main.asc示例代码如下：

```
#include "acl/acl.h"

struct KernelArgs {
    int mode;
};

extern __global__ __aicpu__ uint32_t hello_world(void *args);

int32_t main(int argc, char const *argv[])
{
    aclInit(nullptr);
    int32_t deviceId = 0;
    aclrtSetDevice(deviceId);
    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);

    struct KernelArgs args = {0};
    constexpr uint32_t blockDim = 1;
    hello_world<<<blockDim, nullptr, stream>>>(&args, sizeof(KernelArgs));
    aclrtSynchronizeStream(stream);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```

开发者可以使用bisheng命令行将hello_world.aicpu与main.asc分别编译成.o，再链接成为可执行文件，编译命令如下：

- 编译hello_world.aicpu时，通过-I指定依赖头文件所在路径；通过--cce-aicpu-laicpu_api为Device链接依赖的库libaicpu_api.a，通过--cce-aicpu-L指定libaicpu_api.a的库路径。
- 编译main.asc时，通过--npu-arch编译选项指定对应的架构版本号。

${INSTALL_DIR}请替换为CANN软件安装后文件存储路径。以root用户安装为例，则安装后文件存储路径为：/usr/local/Ascend/cann。

```
$bisheng -O2 hello_world.aicpu --cce-aicpu-L${INSTALL_DIR}/lib64/device/lib64 --cce-aicpu-laicpu_api -I${INSTALL_DIR}/include/ascendc/aicpu_api -c -o hello_world.aicpu.o
$bisheng --npu-arch=dav-2201 main.asc -c -o main.asc.o
$bisheng hello_world.aicpu.o main.asc.o -o demo
```

上文我们通过一个入门示例介绍了使用bisheng命令行编译生成可执行文件的示例。除此之外，使用bisheng命令行也支持编译生成AI CPU算子的动态库与静态库，用户可在asc代码中通过内核调用符<<<...>>>调用AI CPU算子的核函数，并在编译asc代码源文件生成可执行文件的时候，链接AI CPU动态库或者静态库，注意：若单独编译AI CPU算子代码生成动态库、静态库时，需要手动链接表2。

- 编译生成算子动态库

```
# 编译test_aicpu.cpp生成算子动态库
# -lxxx表示默认链接库
# bisheng -shared -x aicpu test_aicpu.cpp -o libtest_aicpu.so -lxxx ...
```

- 编译生成算子静态库

```
# 编译test_aicpu.cpp生成算子静态库
# -lxxx表示默认链接库
# bisheng -lib -x aicpu test_aicpu.cpp -o libtest_aicpu.a -lxxx ...
```

#### AI CPU算子常用编译选项

AI CPU算子常用的编译选项说明如下：

| **选项** | **是否必需** | **说明** |
| --- | --- | --- |
| -help | 否 | 查看帮助。 |
| -x | 否 | 指定编译语言。 指定为aicpu时表示AI CPU算子编程语言。 |
| -o <file> | 否 | 指定输出文件的名称和位置。 |
| -c | 否 | 编译生成目标文件。 |
| -shared，--shared | 否 | 编译生成动态链接库。 |
| -lib | 否 | 编译生成静态链接库。 |
| -g | 否 | 编译时增加调试信息。 |
| -fPIC | 否 | 告知编译器产生位置无关代码。 |
| -O | 否 | 用于指定编译器的优化级别，当前支持-O3，-O2，-O0。 |
| --cce-aicpu-L | 否 | 指定AI CPU Device依赖的库路径。 |
| --cce-aicpu-l | 否 | 指定AI CPU Device依赖的库。 |

#### 通过CMake编译

项目中可以使用CMake来更简便地使用毕昇编译器编译AI CPU算子，生成可执行文件、动态库、静态库或二进制文件。

仍以通过bisheng命令行编译中介绍的Hello World打印样例为例，除了代码实现文件，还需要在工程目录下准备一个CMakeLists.txt。

```
├── hello_world.aicpu // AI CPU算子核函数定义
├── main.asc // AI CPU算子核函数调用
└── CMakeLists.txt
```

CMakeLists.txt内容如下：

```
cmake_minimum_required(VERSION 3.16)
# 1、find_package()是CMake中用于查找和配置Ascend C编译工具链的命令
find_package(ASC REQUIRED) 
find_package(AICPU REQUIRED) 

# 2、指定项目支持的语言包括ASC、AICPU和CXX，ASC表示支持使用毕昇编译器对Ascend C编程语言进行编译，AI CPU表示支持使用毕昇编译器对AI CPU算子进行编译
project(kernel_samples LANGUAGES ASC AICPU CXX)

# 3、使用CMake接口编译可执行文件
add_executable(demo
    hello_world.aicpu
    main.asc
)

#4、由于存在ASC与AI CPU语言，需要指定链接器
set_target_properties(demo PROPERTIES LINKER_LANGUAGE ASC)  //指定链接使用语言

target_include_directories(demo PUBLIC
    ${INSTALL_DIR}/include/ascendc/aicpu_api    
)

target_compile_options(demo PRIVATE
    # --npu-arch用于指定NPU的架构版本，dav-后为架构版本号，各产品型号对应的架构版本号请通过对应关系表进行查询。
    # <COMPILE_LANGUAGE:ASC>:表明该编译选项仅对语言ASC生效
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>
)
```

如果需要CMake编译编译生成动态库、静态库，下面提供了更详细具体的编译示例：

- 编译.cpp文件生成动态库

```
# 将.cpp文件置为ASC属性，启用Ascend C语言进行编译
set_source_files_properties(
    add_custom_base.cpp 
    sub_custom_base.cpp
    PROPERTIES LANGUAGE ASC
)

# 将.cpp文件置为AICPU属性，支持AI CPU算子编译
set_source_files_properties(
    aicpu_kernel.cpp
    PROPERTIES LANGUAGE AICPU
)

add_library(kernel_lib SHARED
    add_custom_base.cpp 
    sub_custom_base.cpp
    aicpu_kernel.cpp # 支持AI CPU算子与AI Core算子一起打包为动态库
)
# AI CPU算子编译时，需要手动链接以下依赖库（若指定链接语言为ASC时，不需要手动链接以下库）
target_link_libraries(kernel_lib PRIVATE
    ascendc_runtime
    profapi
    ascendalog
    ascendcl
    runtime
    c_sec
    mmpa
    error_manager
    ascend_dump
)

add_executable(demo
    main.cpp
)
target_compile_definitions(demo PRIVATE
    ASCENDC_DUMP=0
)
target_compile_options(demo PRIVATE
    -g
)
target_include_directories(demo PRIVATE
    include
)
target_link_libraries(demo PRIVATE
    kernel_lib
)
```

- 编译.asc文件与.aicpu文件生成静态库

```
# .asc文件会默认启用Ascend C语言进行编译，.aicpu文件会默认启用AICPU语言进行编译，不需要通过set_source_files_properties进行设置
add_library(kernel_lib STATIC
    add_custom_base.asc 
    sub_custom_base.asc
    aicpu_kernel.aicpu  # 可支持AI CPU算子与AI Core算子一起打包为静态库
)

add_executable(demo
    main.cpp
)
target_compile_definitions(demo PRIVATE
    ASCENDC_DUMP=0
)
target_compile_options(demo PRIVATE
    -g
)
target_include_directories(demo PRIVATE
    include
)
target_link_libraries(demo PRIVATE
    kernel_lib
)
```

下文列出了使用CMake编译时常用的变量配置说明、常用的链接库。

**表1 **常用的变量配置说明

| 变量 | 配置说明 |
| --- | --- |
| CMAKE_BUILD_TYPE | 编译模式选项，可配置为： - “Release”，Release版本，不包含调试信息，编译最终发布的版本。- “Debug”，Debug版本，包含调试信息，便于开发者开发和调试。 |
| CMAKE_INSTALL_PREFIX | 用于指定CMake执行install时，安装的路径前缀，执行install后编译产物（ascendc_library中指定的target以及对应的头文件）会安装在该路径下。默认路径为当前目录的out目录下。 |
| CMAKE_CXX_COMPILER_LAUNCHER | 用于配置C++语言编译器（如g++）、毕昇编译器的启动器程序为ccache，配置后即可开启cache缓存编译，加速重复编译并提高构建效率。使用该功能前需要安装ccache。 配置方法如下，在对应的CMakeLists.txt进行设置： ``` set(CMAKE_CXX_COMPILER_LAUNCHER <launcher_program>) ``` 其中<launcher_program>是ccache的安装路径，比如ccache的安装路径为/usr/bin/ccache，示例如下： ``` set(CMAKE_CXX_COMPILER_LAUNCHER /usr/bin/ccache) ``` |

**表2 **常用的链接库（在使用高阶API时，必须链接以下库，因为这些库是高阶API功能所依赖的。在其他场景下，可以根据具体需求选择是否链接这些库。）

| 名称 | 作用描述 | 使用场景 |
| --- | --- | --- |
| libtiling_api.a | Tiling函数相关库。 | 使用高阶API相关的Tiling接口时需要链接。 |
| libregister.so | Tiling注册相关库。 | 使用高阶API相关的Tiling接口时需要链接。 |
| libgraph_base.so | 基础数据结构和接口库。 | 调用ge::Shape，ge::DataType等基础结构体时需要链接。 |
| libplatform.so | 硬件平台信息库。 | 使用PlatformAscendC相关硬件平台信息接口时需要链接。 |

**表3 **编译AI CPU算子需要手动链接的库

| 名称 | 作用描述 |
| --- | --- |
| libascendc_runtime.a | Ascend C算子参数等组装库。 |
| libruntime.so | Runtime运行库。 |
| libprofapi.so | Ascend C算子运行性能数据采集库。 |
| libascendalog.so | CANN日志收集库。 |
| libmmpa.so | CANN系统接口库。 |
| libascend_dump.so | CANN维测信息库。 |
| libc_sec.so | CANN安全函数库。 |
| liberror_manager.so | CANN错误信息管理库。 |
| libascendcl.so | acl相关接口库。 |
