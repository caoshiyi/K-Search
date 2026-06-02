# 通过CMake编译

**页面ID:** atlas_ascendc_10_00039  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_10_00039.html

---

项目中可以使用CMake来更简便地使用毕昇编译器编译Ascend C算子，生成可执行文件、动态库、静态库或二进制文件。

以下是CMake脚本的示例及其核心步骤说明：

```
# 1、find_package(ASC)是CMake中用于查找和配置Ascend C编译工具链的命令
find_package(ASC)  

# 2、指定项目支持的语言包括ASC和CXX，ASC表示支持使用毕昇编译器对Ascend C编程语言进行编译
project(kernel_samples LANGUAGES ASC CXX)

# 3、使用CMake接口编译可执行文件、动态库、静态库、二进制文件
add_executable(demo
    add_custom.asc
)
#.....
target_compile_options(demo PRIVATE
    # --npu-arch用于指定NPU的架构版本，dav-后为架构版本号，各产品型号对应的架构版本号请通过对应关系表进行查询。
    # <COMPILE_LANGUAGE:ASC>:表明该编译选项仅对语言ASC生效
    $<$<COMPILE_LANGUAGE:ASC>: --npu-arch=dav-2201>    
)
```

以下是动态库、静态库编译示例，同时展示如何将源文件切换为用语言ASC编译：

- 编译.cpp文件生成动态库

```
# 将.cpp文件置为ASC属性，启用Ascend C语言进行编译
set_source_files_properties(
    add_custom_base.cpp 
    sub_custom_base.cpp
    PROPERTIES LANGUAGE ASC
)

add_library(kernel_lib SHARED
    add_custom_base.cpp 
    sub_custom_base.cpp
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

- 编译.asc文件生成静态库

```
# .asc文件会默认启用Ascend C语言进行编译，不需要通过set_source_files_properties进行设置
add_library(kernel_lib STATIC
    add_custom_base.asc 
    sub_custom_base.asc
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

下文列出了使用CMake编译时常用的链接库、以及默认链接库。

**表1 **常用的链接库（在使用高阶API时，必须链接以下库，因为这些库是高阶API功能所依赖的。在其他场景下，可以根据具体需求选择是否链接这些库。）

| 名称 | 作用描述 | 使用场景 |
| --- | --- | --- |
| libtiling_api.a | Tiling函数相关库。 | 使用高阶API相关的Tiling接口时需要链接。 |
| libregister.so | Tiling注册相关库。 | 使用高阶API相关的Tiling接口时需要链接。 |
| libgraph_base.so | 基础数据结构和接口库。 | 调用ge::Shape，ge::DataType等基础结构体时需要链接。 |
| libplatform.so | 硬件平台信息库。 | 使用PlatformAscendC相关硬件平台信息接口时需要链接。 |

**表2 **默认链接库

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
