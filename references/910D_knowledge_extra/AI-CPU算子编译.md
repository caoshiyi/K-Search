# AI CPU算子编译<a name="ZH-CN_TOPIC_0000002554431445"></a>

## 通过bisheng命令行编译<a name="section153291123460"></a>

下文基于一个Hello World打印样例来讲解如何通过bisheng命令行编译AI CPU算子。

hello\_world.aicpu文件内容如下：

```
#include "aicpu_api.h"

__global__ __aicpu__ uint32_t hello_world(void *args)
{
    AscendC::printf("Hello World!!!\n");
    return 0;
}
```

Host侧使用内核调用符<<<...\>\>\>进行AI CPU算子的调用， main.asc示例代码如下：

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
    constexpr uint32_t numBlocks = 1;
    hello_world<<<numBlocks, nullptr, stream>>>(&args, sizeof(KernelArgs));
    aclrtSynchronizeStream(stream);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```

开发者可以使用bisheng命令行将hello\_world.aicpu与main.asc分别编译成.o，再链接成为可执行文件，编译命令如下：

-   编译hello\_world.aicpu时，通过-I指定依赖头文件所在路径；通过--cce-aicpu-laicpu\_api为Device链接依赖的库libaicpu\_api.a，通过--cce-aicpu-L指定libaicpu\_api.a的库路径。
-   编译main.asc时，通过--npu-arch编译选项指定对应的架构版本号。

$\{INSTALL\_DIR\}请替换为CANN软件安装后文件存储路径。以root用户安装为例，安装后文件默认存储路径为：/usr/local/Ascend/cann。

```
$bisheng -O2 hello_world.aicpu --cce-aicpu-L${INSTALL_DIR}/lib64/device/lib64 --cce-aicpu-laicpu_api -I${INSTALL_DIR}/include/ascendc/aicpu_api -c -o hello_world.aicpu.o
# --npu-arch用于指定NPU的架构版本，dav-后为架构版本号，各产品型号对应的架构版本号请通过[对应关系表](SIMD-BuiltIn关键字和API.md#table65291052154114)进行查询。
$bisheng --npu-arch=dav-2201 main.asc -c -o main.asc.o
$bisheng hello_world.aicpu.o main.asc.o -o demo
```

上文我们通过一个入门示例介绍了使用bisheng命令行编译生成可执行文件的示例。除此之外，使用bisheng命令行也支持编译生成AI CPU算子的动态库与静态库，用户可在asc代码中通过内核调用符<<<...\>\>\>调用AI CPU算子的核函数，并在编译asc代码源文件生成可执行文件的时候，链接AI CPU动态库或者静态库，注意：若单独编译AI CPU算子代码生成动态库、静态库时，需要手动链接[表2](通过CMake编译.md#table201231542115513)。

-   编译生成算子动态库

    ```
    # 编译test_aicpu.cpp生成算子动态库
    # -lxxx表示默认链接库
    # bisheng -shared -x aicpu test_aicpu.cpp -o libtest_aicpu.so -lxxx ...
    ```

-   编译生成算子静态库

    ```
    # 编译test_aicpu.cpp生成算子静态库
    # -lxxx表示默认链接库
    # bisheng -lib -x aicpu test_aicpu.cpp -o libtest_aicpu.a -lxxx ...
    ```

## AI CPU算子常用编译选项<a name="section345885113142"></a>

AI CPU算子常用的编译选项说明如下：

<a name="table9126181131320"></a>
<table><thead align="left"><tr id="row312711101316"><th class="cellrowborder" valign="top" width="33.63636363636363%" id="mcps1.1.4.1.1"><p id="p71271711201318"><a name="p71271711201318"></a><a name="p71271711201318"></a><strong id="b01279110139"><a name="b01279110139"></a><a name="b01279110139"></a>选项</strong></p>
</th>
<th class="cellrowborder" valign="top" width="9.676767676767676%" id="mcps1.1.4.1.2"><p id="p1212711115131"><a name="p1212711115131"></a><a name="p1212711115131"></a><strong id="b101271011101310"><a name="b101271011101310"></a><a name="b101271011101310"></a>是否必需</strong></p>
</th>
<th class="cellrowborder" valign="top" width="56.686868686868685%" id="mcps1.1.4.1.3"><p id="p8127121151311"><a name="p8127121151311"></a><a name="p8127121151311"></a><strong id="b15127191120134"><a name="b15127191120134"></a><a name="b15127191120134"></a>说明</strong></p>
</th>
</tr>
</thead>
<tbody><tr id="row8127161113135"><td class="cellrowborder" valign="top" width="33.63636363636363%" headers="mcps1.1.4.1.1 "><p id="p131279114139"><a name="p131279114139"></a><a name="p131279114139"></a>-help</p>
</td>
<td class="cellrowborder" valign="top" width="9.676767676767676%" headers="mcps1.1.4.1.2 "><p id="p61271711121318"><a name="p61271711121318"></a><a name="p61271711121318"></a>否</p>
</td>
<td class="cellrowborder" valign="top" width="56.686868686868685%" headers="mcps1.1.4.1.3 "><p id="p14127201181312"><a name="p14127201181312"></a><a name="p14127201181312"></a>查看帮助。</p>
</td>
</tr>
<tr id="row19128611141312"><td class="cellrowborder" valign="top" width="33.63636363636363%" headers="mcps1.1.4.1.1 "><p id="p17128511131312"><a name="p17128511131312"></a><a name="p17128511131312"></a>-x</p>
</td>
<td class="cellrowborder" valign="top" width="9.676767676767676%" headers="mcps1.1.4.1.2 "><p id="p13128181141318"><a name="p13128181141318"></a><a name="p13128181141318"></a>否</p>
</td>
<td class="cellrowborder" valign="top" width="56.686868686868685%" headers="mcps1.1.4.1.3 "><p id="p4128101191320"><a name="p4128101191320"></a><a name="p4128101191320"></a>指定编译语言。</p>
<p id="p21281116136"><a name="p21281116136"></a><a name="p21281116136"></a>指定为aicpu时表示AI CPU算子编程语言。</p>
</td>
</tr>
<tr id="row10128111115130"><td class="cellrowborder" valign="top" width="33.63636363636363%" headers="mcps1.1.4.1.1 "><p id="p3128191110133"><a name="p3128191110133"></a><a name="p3128191110133"></a>-o &lt;file&gt;</p>
</td>
<td class="cellrowborder" valign="top" width="9.676767676767676%" headers="mcps1.1.4.1.2 "><p id="p181287119131"><a name="p181287119131"></a><a name="p181287119131"></a>否</p>
</td>
<td class="cellrowborder" valign="top" width="56.686868686868685%" headers="mcps1.1.4.1.3 "><p id="p812861131314"><a name="p812861131314"></a><a name="p812861131314"></a>指定输出文件的名称和位置。</p>
</td>
</tr>
<tr id="row7128911121316"><td class="cellrowborder" valign="top" width="33.63636363636363%" headers="mcps1.1.4.1.1 "><p id="p2012821151310"><a name="p2012821151310"></a><a name="p2012821151310"></a>-c</p>
</td>
<td class="cellrowborder" valign="top" width="9.676767676767676%" headers="mcps1.1.4.1.2 "><p id="p11128151116138"><a name="p11128151116138"></a><a name="p11128151116138"></a>否</p>
</td>
<td class="cellrowborder" valign="top" width="56.686868686868685%" headers="mcps1.1.4.1.3 "><p id="p131281311121315"><a name="p131281311121315"></a><a name="p131281311121315"></a>编译生成目标文件。</p>
</td>
</tr>
<tr id="row15128151111314"><td class="cellrowborder" valign="top" width="33.63636363636363%" headers="mcps1.1.4.1.1 "><p id="p1312831111136"><a name="p1312831111136"></a><a name="p1312831111136"></a>-shared，--shared</p>
</td>
<td class="cellrowborder" valign="top" width="9.676767676767676%" headers="mcps1.1.4.1.2 "><p id="p91289111133"><a name="p91289111133"></a><a name="p91289111133"></a>否</p>
</td>
<td class="cellrowborder" valign="top" width="56.686868686868685%" headers="mcps1.1.4.1.3 "><p id="p10128171112137"><a name="p10128171112137"></a><a name="p10128171112137"></a>编译生成动态链接库。</p>
</td>
</tr>
<tr id="row512881114134"><td class="cellrowborder" valign="top" width="33.63636363636363%" headers="mcps1.1.4.1.1 "><p id="p1912841161317"><a name="p1912841161317"></a><a name="p1912841161317"></a>-lib</p>
</td>
<td class="cellrowborder" valign="top" width="9.676767676767676%" headers="mcps1.1.4.1.2 "><p id="p212813114130"><a name="p212813114130"></a><a name="p212813114130"></a>否</p>
</td>
<td class="cellrowborder" valign="top" width="56.686868686868685%" headers="mcps1.1.4.1.3 "><p id="p612811111131"><a name="p612811111131"></a><a name="p612811111131"></a>编译生成静态链接库。</p>
</td>
</tr>
<tr id="row1912891101318"><td class="cellrowborder" valign="top" width="33.63636363636363%" headers="mcps1.1.4.1.1 "><p id="p81287111130"><a name="p81287111130"></a><a name="p81287111130"></a>-g</p>
</td>
<td class="cellrowborder" valign="top" width="9.676767676767676%" headers="mcps1.1.4.1.2 "><p id="p1212811115139"><a name="p1212811115139"></a><a name="p1212811115139"></a>否</p>
</td>
<td class="cellrowborder" valign="top" width="56.686868686868685%" headers="mcps1.1.4.1.3 "><p id="p41281811131314"><a name="p41281811131314"></a><a name="p41281811131314"></a>编译时增加调试信息。</p>
</td>
</tr>
<tr id="row1128911201315"><td class="cellrowborder" valign="top" width="33.63636363636363%" headers="mcps1.1.4.1.1 "><p id="p2128161131317"><a name="p2128161131317"></a><a name="p2128161131317"></a>-fPIC</p>
</td>
<td class="cellrowborder" valign="top" width="9.676767676767676%" headers="mcps1.1.4.1.2 "><p id="p112871121316"><a name="p112871121316"></a><a name="p112871121316"></a>否</p>
</td>
<td class="cellrowborder" valign="top" width="56.686868686868685%" headers="mcps1.1.4.1.3 "><p id="p1912817114131"><a name="p1912817114131"></a><a name="p1912817114131"></a>告知编译器产生位置无关代码。</p>
</td>
</tr>
<tr id="row3128151113131"><td class="cellrowborder" valign="top" width="33.63636363636363%" headers="mcps1.1.4.1.1 "><p id="p312831171319"><a name="p312831171319"></a><a name="p312831171319"></a>-O</p>
</td>
<td class="cellrowborder" valign="top" width="9.676767676767676%" headers="mcps1.1.4.1.2 "><p id="p14128131114137"><a name="p14128131114137"></a><a name="p14128131114137"></a>否</p>
</td>
<td class="cellrowborder" valign="top" width="56.686868686868685%" headers="mcps1.1.4.1.3 "><p id="p1412851101319"><a name="p1412851101319"></a><a name="p1412851101319"></a>用于指定编译器的优化级别，当前支持-O3，-O2，-O0。</p>
</td>
</tr>
<tr id="row118491817141416"><td class="cellrowborder" valign="top" width="33.63636363636363%" headers="mcps1.1.4.1.1 "><p id="p1565513583296"><a name="p1565513583296"></a><a name="p1565513583296"></a>--cce-aicpu-L</p>
</td>
<td class="cellrowborder" valign="top" width="9.676767676767676%" headers="mcps1.1.4.1.2 "><p id="p2655195822910"><a name="p2655195822910"></a><a name="p2655195822910"></a>否</p>
</td>
<td class="cellrowborder" valign="top" width="56.686868686868685%" headers="mcps1.1.4.1.3 "><p id="p365515813296"><a name="p365515813296"></a><a name="p365515813296"></a>指定AI CPU Device依赖的库路径。</p>
</td>
</tr>
<tr id="row49581340171415"><td class="cellrowborder" valign="top" width="33.63636363636363%" headers="mcps1.1.4.1.1 "><p id="p196884185304"><a name="p196884185304"></a><a name="p196884185304"></a>--cce-aicpu-l</p>
</td>
<td class="cellrowborder" valign="top" width="9.676767676767676%" headers="mcps1.1.4.1.2 "><p id="p17688161833011"><a name="p17688161833011"></a><a name="p17688161833011"></a>否</p>
</td>
<td class="cellrowborder" valign="top" width="56.686868686868685%" headers="mcps1.1.4.1.3 "><p id="p1688918113011"><a name="p1688918113011"></a><a name="p1688918113011"></a>指定AI CPU Device依赖的库。</p>
</td>
</tr>
</tbody>
</table>

## 通过CMake编译<a name="section1121825118533"></a>

项目中可以使用CMake来更简便地使用毕昇编译器编译AI CPU算子，生成可执行文件、动态库、静态库或二进制文件。

仍以[通过bisheng命令行编译](#section153291123460)中介绍的Hello World打印样例为例，除了代码实现文件，还需要在工程目录下准备一个CMakeLists.txt。

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
    # --npu-arch用于指定NPU的架构版本，dav-后为架构版本号，各产品型号对应的架构版本号请通过[对应关系表](SIMD-BuiltIn关键字和API.md#table65291052154114)进行查询。
    # <COMPILE_LANGUAGE:ASC>:表明该编译选项仅对语言ASC生效
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>
)
```

如果需要CMake编译编译生成动态库、静态库，下面提供了更详细具体的编译示例：

-   编译.cpp文件生成动态库

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
        unified_dlog
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

-   编译.asc文件与.aicpu文件生成静态库

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

**表 1**  常用的变量配置说明

<a name="table244174535419"></a>
<table><thead align="left"><tr id="row17451145145411"><th class="cellrowborder" valign="top" width="24.54%" id="mcps1.2.3.1.1"><p id="p18451345115414"><a name="p18451345115414"></a><a name="p18451345115414"></a>变量</p>
</th>
<th class="cellrowborder" valign="top" width="75.46000000000001%" id="mcps1.2.3.1.2"><p id="p4451245115415"><a name="p4451245115415"></a><a name="p4451245115415"></a>配置说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row194514510541"><td class="cellrowborder" valign="top" width="24.54%" headers="mcps1.2.3.1.1 "><p id="p04514519549"><a name="p04514519549"></a><a name="p04514519549"></a>CMAKE_BUILD_TYPE</p>
</td>
<td class="cellrowborder" valign="top" width="75.46000000000001%" headers="mcps1.2.3.1.2 "><p id="p6451456545"><a name="p6451456545"></a><a name="p6451456545"></a>编译模式选项，可配置为：</p>
<a name="ul145124517543"></a><a name="ul145124517543"></a><ul id="ul145124517543"><li>“Release”，Release版本，不包含调试信息，编译最终发布的版本。</li><li>“Debug”，Debug版本，包含调试信息，便于开发者开发和调试。</li></ul>
</td>
</tr>
<tr id="row74518458542"><td class="cellrowborder" valign="top" width="24.54%" headers="mcps1.2.3.1.1 "><p id="p14520459540"><a name="p14520459540"></a><a name="p14520459540"></a>CMAKE_INSTALL_PREFIX</p>
</td>
<td class="cellrowborder" valign="top" width="75.46000000000001%" headers="mcps1.2.3.1.2 "><p id="p145114515419"><a name="p145114515419"></a><a name="p145114515419"></a>用于指定CMake执行install时，安装的路径前缀，执行install后编译产物（ascendc_library中指定的target以及对应的头文件）会安装在该路径下。默认路径为当前目录的out目录下。</p>
</td>
</tr>
<tr id="row124534535416"><td class="cellrowborder" valign="top" width="24.54%" headers="mcps1.2.3.1.1 "><p id="p144516455548"><a name="p144516455548"></a><a name="p144516455548"></a>CMAKE_CXX_COMPILER_LAUNCHER</p>
</td>
<td class="cellrowborder" valign="top" width="75.46000000000001%" headers="mcps1.2.3.1.2 "><p id="p845134505416"><a name="p845134505416"></a><a name="p845134505416"></a>用于配置C++语言编译器（如g++）、毕昇编译器的启动器程序为ccache，配置后即可开启cache缓存编译，<span>加速重复编译并提高构建效率</span>。使用该功能前需要安装ccache。</p>
<p id="p134516454544"><a name="p134516454544"></a><a name="p134516454544"></a>配置方法如下，在对应的CMakeLists.txt进行设置：</p>
<pre class="screen" id="screen4456455542"><a name="screen4456455542"></a><a name="screen4456455542"></a>set(CMAKE_CXX_COMPILER_LAUNCHER &lt;launcher_program&gt;)</pre>
<p id="p1545164515419"><a name="p1545164515419"></a><a name="p1545164515419"></a>其中&lt;launcher_program&gt;是ccache的安装路径，比如ccache的安装路径为/usr/bin/ccache，示例如下：</p>
<pre class="screen" id="screen245345155418"><a name="screen245345155418"></a><a name="screen245345155418"></a>set(CMAKE_CXX_COMPILER_LAUNCHER /usr/bin/ccache)</pre>
</td>
</tr>
</tbody>
</table>

**表 2**  常用的链接库（在使用高阶API时，必须链接以下库，因为这些库是高阶API功能所依赖的。在其他场景下，可以根据具体需求选择是否链接这些库。）

<a name="table545745125417"></a>
<table><thead align="left"><tr id="row3461457541"><th class="cellrowborder" valign="top" width="24.099999999999998%" id="mcps1.2.4.1.1"><p id="p446174535418"><a name="p446174535418"></a><a name="p446174535418"></a>名称</p>
</th>
<th class="cellrowborder" valign="top" width="33.019999999999996%" id="mcps1.2.4.1.2"><p id="p174684595413"><a name="p174684595413"></a><a name="p174684595413"></a>作用描述</p>
</th>
<th class="cellrowborder" valign="top" width="42.88%" id="mcps1.2.4.1.3"><p id="p6463459549"><a name="p6463459549"></a><a name="p6463459549"></a>使用场景</p>
</th>
</tr>
</thead>
<tbody><tr id="row5465454546"><td class="cellrowborder" valign="top" width="24.099999999999998%" headers="mcps1.2.4.1.1 "><p id="p154611454540"><a name="p154611454540"></a><a name="p154611454540"></a>libtiling_api.a</p>
</td>
<td class="cellrowborder" valign="top" width="33.019999999999996%" headers="mcps1.2.4.1.2 "><p id="p134611450541"><a name="p134611450541"></a><a name="p134611450541"></a>Tiling函数相关库。</p>
</td>
<td class="cellrowborder" valign="top" width="42.88%" headers="mcps1.2.4.1.3 "><p id="p194634510544"><a name="p194634510544"></a><a name="p194634510544"></a>使用高阶API相关的Tiling接口时需要链接。</p>
</td>
</tr>
<tr id="row246945115419"><td class="cellrowborder" valign="top" width="24.099999999999998%" headers="mcps1.2.4.1.1 "><p id="p1946124575411"><a name="p1946124575411"></a><a name="p1946124575411"></a>libregister.so</p>
</td>
<td class="cellrowborder" valign="top" width="33.019999999999996%" headers="mcps1.2.4.1.2 "><p id="p19461945115413"><a name="p19461945115413"></a><a name="p19461945115413"></a>Tiling注册相关库。</p>
</td>
<td class="cellrowborder" valign="top" width="42.88%" headers="mcps1.2.4.1.3 "><p id="p24664515417"><a name="p24664515417"></a><a name="p24664515417"></a>使用高阶API相关的Tiling接口时需要链接。</p>
</td>
</tr>
<tr id="row046245115413"><td class="cellrowborder" valign="top" width="24.099999999999998%" headers="mcps1.2.4.1.1 "><p id="p19466459541"><a name="p19466459541"></a><a name="p19466459541"></a>libgraph_base.so</p>
</td>
<td class="cellrowborder" valign="top" width="33.019999999999996%" headers="mcps1.2.4.1.2 "><p id="p19461445125419"><a name="p19461445125419"></a><a name="p19461445125419"></a>基础数据结构和接口库。</p>
</td>
<td class="cellrowborder" valign="top" width="42.88%" headers="mcps1.2.4.1.3 "><p id="p24674535412"><a name="p24674535412"></a><a name="p24674535412"></a>调用ge::Shape，ge::DataType等基础结构体时需要链接。</p>
</td>
</tr>
<tr id="row154634513544"><td class="cellrowborder" valign="top" width="24.099999999999998%" headers="mcps1.2.4.1.1 "><p id="p114644575413"><a name="p114644575413"></a><a name="p114644575413"></a>libplatform.so</p>
</td>
<td class="cellrowborder" valign="top" width="33.019999999999996%" headers="mcps1.2.4.1.2 "><p id="p446184512545"><a name="p446184512545"></a><a name="p446184512545"></a>硬件平台信息库。</p>
</td>
<td class="cellrowborder" valign="top" width="42.88%" headers="mcps1.2.4.1.3 "><p id="p1346645165417"><a name="p1346645165417"></a><a name="p1346645165417"></a>使用PlatformAscendC相关硬件平台信息接口时需要链接。</p>
</td>
</tr>
</tbody>
</table>

**表 3**  编译AI CPU算子需要手动链接的库

<a name="table346945145419"></a>
<table><thead align="left"><tr id="row247245105412"><th class="cellrowborder" valign="top" width="23.98%" id="mcps1.2.3.1.1"><p id="p247194525413"><a name="p247194525413"></a><a name="p247194525413"></a>名称</p>
</th>
<th class="cellrowborder" valign="top" width="76.02%" id="mcps1.2.3.1.2"><p id="p847154515416"><a name="p847154515416"></a><a name="p847154515416"></a>作用描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row1747184585416"><td class="cellrowborder" valign="top" width="23.98%" headers="mcps1.2.3.1.1 "><p id="p1047154517544"><a name="p1047154517544"></a><a name="p1047154517544"></a>libascendc_runtime.a</p>
</td>
<td class="cellrowborder" valign="top" width="76.02%" headers="mcps1.2.3.1.2 "><p id="p1647545185419"><a name="p1647545185419"></a><a name="p1647545185419"></a>Ascend C算子参数等组装库。</p>
</td>
</tr>
<tr id="row2471459548"><td class="cellrowborder" valign="top" width="23.98%" headers="mcps1.2.3.1.1 "><p id="p194724511542"><a name="p194724511542"></a><a name="p194724511542"></a>libruntime.so</p>
</td>
<td class="cellrowborder" valign="top" width="76.02%" headers="mcps1.2.3.1.2 "><p id="p20471845185417"><a name="p20471845185417"></a><a name="p20471845185417"></a>Runtime运行库。</p>
</td>
</tr>
<tr id="row124734515546"><td class="cellrowborder" valign="top" width="23.98%" headers="mcps1.2.3.1.1 "><p id="p147134512548"><a name="p147134512548"></a><a name="p147134512548"></a>libprofapi.so</p>
</td>
<td class="cellrowborder" valign="top" width="76.02%" headers="mcps1.2.3.1.2 "><p id="p14719452541"><a name="p14719452541"></a><a name="p14719452541"></a>Ascend C算子运行性能数据采集库。</p>
</td>
</tr>
<tr id="row1647345105420"><td class="cellrowborder" valign="top" width="23.98%" headers="mcps1.2.3.1.1 "><p id="p104784517542"><a name="p104784517542"></a><a name="p104784517542"></a>libunified_dlog.so</p>
</td>
<td class="cellrowborder" valign="top" width="76.02%" headers="mcps1.2.3.1.2 "><p id="p147184545410"><a name="p147184545410"></a><a name="p147184545410"></a>CANN日志收集库。</p>
</td>
</tr>
<tr id="row54717452542"><td class="cellrowborder" valign="top" width="23.98%" headers="mcps1.2.3.1.1 "><p id="p1247124511546"><a name="p1247124511546"></a><a name="p1247124511546"></a>libmmpa.so</p>
</td>
<td class="cellrowborder" valign="top" width="76.02%" headers="mcps1.2.3.1.2 "><p id="p34774520544"><a name="p34774520544"></a><a name="p34774520544"></a>CANN系统接口库。</p>
</td>
</tr>
<tr id="row247114520549"><td class="cellrowborder" valign="top" width="23.98%" headers="mcps1.2.3.1.1 "><p id="p8471645165412"><a name="p8471645165412"></a><a name="p8471645165412"></a>libascend_dump.so</p>
</td>
<td class="cellrowborder" valign="top" width="76.02%" headers="mcps1.2.3.1.2 "><p id="p1947164513547"><a name="p1947164513547"></a><a name="p1947164513547"></a>CANN维测信息库。</p>
</td>
</tr>
<tr id="row747114515415"><td class="cellrowborder" valign="top" width="23.98%" headers="mcps1.2.3.1.1 "><p id="p6473454549"><a name="p6473454549"></a><a name="p6473454549"></a>libc_sec.so</p>
</td>
<td class="cellrowborder" valign="top" width="76.02%" headers="mcps1.2.3.1.2 "><p id="p1847204512547"><a name="p1847204512547"></a><a name="p1847204512547"></a>CANN安全函数库。</p>
</td>
</tr>
<tr id="row3472045145410"><td class="cellrowborder" valign="top" width="23.98%" headers="mcps1.2.3.1.1 "><p id="p124819454540"><a name="p124819454540"></a><a name="p124819454540"></a>liberror_manager.so</p>
</td>
<td class="cellrowborder" valign="top" width="76.02%" headers="mcps1.2.3.1.2 "><p id="p2484452543"><a name="p2484452543"></a><a name="p2484452543"></a>CANN错误信息管理库。</p>
</td>
</tr>
<tr id="row7487453546"><td class="cellrowborder" valign="top" width="23.98%" headers="mcps1.2.3.1.1 "><p id="p84884585415"><a name="p84884585415"></a><a name="p84884585415"></a>libascendcl.so</p>
</td>
<td class="cellrowborder" valign="top" width="76.02%" headers="mcps1.2.3.1.2 "><p id="p124817452541"><a name="p124817452541"></a><a name="p124817452541"></a>acl相关接口库。</p>
</td>
</tr>
</tbody>
</table>

