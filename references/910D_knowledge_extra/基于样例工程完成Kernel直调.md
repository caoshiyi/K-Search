# 基于样例工程完成Kernel直调<a name="ZH-CN_TOPIC_0000002523351586"></a>

> **说明：** 
>本章节介绍的基于样例工程完成Kernel直调的方式，后续不再演进。推荐开发者直接使用命令行或者编写Cmake文件进行编译，详细内容请参考[AI Core算子编译](AI-Core算子编译.md)。

下文将以Add矢量算子为例对Kernel直调算子开发流程进行详细介绍。

更多算子样例工程请通过如下链接获取：

-   [矢量算子样例](https://gitee.com/ascend/samples/tree/master/operator/ascendc/0_introduction/3_add_kernellaunch/AddKernelInvocationNeo)
-   [支持Tiling的矢量算子样例](https://gitee.com/ascend/samples/tree/master/operator/ascendc/0_introduction/3_add_kernellaunch/AddKernelInvocationTilingNeo)
-   [矩阵算子样例](https://gitee.com/ascend/samples/tree/master/operator/ascendc/0_introduction/11_matmul_kernellaunch/MatmulInvocationNeo)
-   [矢量+矩阵融合算子样例](https://gitee.com/ascend/samples/tree/master/operator/ascendc/0_introduction/13_matmulleakyrelu_kernellaunch/MatmulLeakyReluInvocation)

## 环境准备<a name="section33636642419"></a>

-   使用Kernel Launch算子工程之前，需要参考[环境准备](环境准备.md)章节安装驱动固件和CANN软件包，完成开发环境和运行环境的准备。
-   使用该算子工程要求cmake版本为3.16及以上版本，如不符合要求，请参考如下的命令示例更新cmake版本，如下示例以更新到3.16.0版本为例。

    ```
    wget https://cmake.org/files/v3.16/cmake-3.16.0.tar.gz 
    tar -zxvf cmake-3.16.0.tar.gz 
    cd cmake-3.16.0
     ./bootstrap --prefix=/usr 
    sudo make 
    sudo make install
    ```

## 工程目录<a name="section7363202715148"></a>

您可以单击[矢量算子样例](https://gitee.com/ascend/samples/tree/master/operator/ascendc/0_introduction/3_add_kernellaunch/AddKernelInvocationNeo)，获取核函数开发和运行验证的完整样例。样例目录结构如下所示：

```
AddKernelInvocationNeo
|-- cmake                                                 // CMake编译文件
|-- scripts
|  ├── gen_data.py                                     // 输入数据和真值数据生成脚本文件
|  ├── verify_result.py                                // 验证输出数据和真值数据是否一致的验证脚本
|-- CMakeLists.txt                                        // CMake编译配置文件
|-- add_custom.cpp                                        // 矢量算子kernel实现
|-- data_utils.h                                          // 数据读入写出函数
|-- main.cpp                                              // 主函数，调用算子的应用程序，含CPU域及NPU域调用
|-- run.sh                                                // 编译运行算子的脚本
```

基于该算子工程，开发者进行算子开发的步骤如下：

-   完成算子kernel侧实现。
-   编写算子调用应用程序main.cpp。
-   编写CMake编译配置文件CMakeLists.txt。

-   根据实际需要修改输入数据和真值数据生成脚本文件gen\_data.py；验证输出数据和真值数据是否一致的验证脚本verify\_result.py。
-   根据实际需要修改编译运行算子的脚本run.sh并执行该脚本，完成算子的编译运行和结果验证。

## 算子Kernel侧实现<a name="section186929199494"></a>

请参考工程目录中的矢量算子、矩阵算子、融合算子的Kernel实现完成Ascend C算子实现文件的编写。

> **说明：** 
>一个算子Kernel实现文件中只支持定义一个核函数。

## 算子调用应用程序<a name="section883611324486"></a>

下面代码以固定shape的add\_custom算子为例，介绍算子核函数调用的应用程序main.cpp如何编写。您在实现自己的应用程序时，需要关注由于算子核函数不同带来的修改，包括算子核函数名，入参出参的不同等，合理安排相应的内存分配、内存拷贝和文件读写等，相关API的调用方式直接复用即可。

1.  按需包含头文件，通过ASCENDC\_CPU\_DEBUG宏区分CPU/NPU侧需要包含的头文件。需要注意的是，NPU侧需要包含对应的核函数调用接口声明所在的头文件aclrtlaunch\_\{kernel\_name\}.h（该头文件为工程框架自动生成），kernel\_name为算子核函数的名称。

    ```
    #include "data_utils.h"
    #ifndef ASCENDC_CPU_DEBUG
    #include "acl/acl.h"
    #include "aclrtlaunch_add_custom.h"
    #else
    #include "tikicpulib.h"
    extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z);
    #endif
    ```

2.  应用程序框架编写。该应用程序通过ASCENDC\_CPU\_DEBUG宏区分代码逻辑运行于CPU侧还是NPU侧。

    ```
    int32_t main(int32_t argc, char* argv[])
    {
        uint32_t numBlocks = 8;
        size_t inputByteSize = 8 * 2048 * sizeof(uint16_t);
        size_t outputByteSize = 8 * 2048 * sizeof(uint16_t);
    
    #ifdef ASCENDC_CPU_DEBUG
        // 用于CPU调试的调用程序
    #else
        // NPU侧运行算子的调用程序
    #endif
        return 0;
    }
    ```

3.  CPU侧运行验证。完成算子核函数CPU侧运行验证的步骤如下：

    **图 1**  CPU侧运行验证步骤<a name="fig13576112114442"></a>  
    <!-- img2text -->
```text
┌────────────────────────────────────┐
│ 使用GmAlloc分配共享内存，并进行    │
│ 数据初始化                         │
└────────────────────────────────────┘
                  │
                  ▼
┌────────────────────────────────────┐
│ 调用CPU_RUN_KF调测宏，完成         │
│ 核函数CPU侧的调用                  │
└────────────────────────────────────┘
                  │
                  ▼
┌────────────────────────────────────┐
│ 输出数据写出                       │
└────────────────────────────────────┘
                  │
                  ▼
┌────────────────────────────────────┐
│ 调用GmFree释放申请的资源           │
└────────────────────────────────────┘
```

    ```
        // 使用GmAlloc分配共享内存，并进行数据初始化
        uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
        uint8_t* y = (uint8_t*)AscendC::GmAlloc(inputByteSize);
        uint8_t* z = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    
        ReadFile("./input/input_x.bin", inputByteSize, x, inputByteSize);
        ReadFile("./input/input_y.bin", inputByteSize, y, inputByteSize);
        // 矢量算子需要设置内核模式为AIV模式
        AscendC::SetKernelMode(KernelMode::AIV_MODE);
        // 调用ICPU_RUN_KF调测宏，完成核函数CPU侧的调用
        ICPU_RUN_KF(add_custom, numBlocks, x, y, z);
        // 输出数据写出
        WriteFile("./output/output_z.bin", z, outputByteSize);
        // 调用GmFree释放申请的资源
        AscendC::GmFree((void *)x);
        AscendC::GmFree((void *)y);
        AscendC::GmFree((void *)z);
    ```

4.  NPU侧运行验证。完成算子核函数NPU侧运行验证的步骤如下：

    **图 2**  NPU侧运行验证步骤<a name="fig558132018817"></a>  
    <!-- img2text -->
```text
┌──────────────────────────┐
│      AscendCL初始化      │
└──────────────────────────┘
            │
            ▼
┌──────────────────────────┐
│      运行管理资源申请    │
└──────────────────────────┘
            │
            ▼
┌──────────────────────────┐
│ 分配Host内存，并进行数据 │
│        初始化            │
└──────────────────────────┘
            │
            ▼
┌──────────────────────────┐
│ 分配Device内存，并将数据 │
│ 从Host上拷贝到Device上   │
└──────────────────────────┘
            │
            ▼
┌──────────────────────────┐
│ 通过Kernel Launch接口或内核 │
│ 调用符<<<>>>调用核函数完成 │
│ 指定的运算，并同步等待   │
└──────────────────────────┘
            │
            ▼
┌──────────────────────────┐
│ 将Device上的运算结果     │
│       拷贝回Host         │
└──────────────────────────┘
            │
            ▼
┌──────────────────────────┐
│      释放申请的资源      │
└──────────────────────────┘
            │
            ▼
┌──────────────────────────┐
│      AscendCL去初始化    │
└──────────────────────────┘
```

    ```
        // 初始化
        CHECK_ACL(aclInit(nullptr));
        // 运行管理资源申请
        int32_t deviceId = 0;
        CHECK_ACL(aclrtSetDevice(deviceId));
        aclrtStream stream = nullptr;
        CHECK_ACL(aclrtCreateStream(&stream));
        // 分配Host内存
        uint8_t *xHost, *yHost, *zHost;
        uint8_t *xDevice, *yDevice, *zDevice;
    
        CHECK_ACL(aclrtMallocHost((void**)(&xHost), inputByteSize));
        CHECK_ACL(aclrtMallocHost((void**)(&yHost), inputByteSize));
        CHECK_ACL(aclrtMallocHost((void**)(&zHost), outputByteSize));
        // 分配Device内存
        CHECK_ACL(aclrtMalloc((void**)&xDevice, inputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
        CHECK_ACL(aclrtMalloc((void**)&yDevice, inputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
        CHECK_ACL(aclrtMalloc((void**)&zDevice, outputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
        // Host内存初始化
        ReadFile("./input/input_x.bin", inputByteSize, xHost, inputByteSize);
        ReadFile("./input/input_y.bin", inputByteSize, yHost, inputByteSize);
        // 将数据从Host上拷贝到Device上
        CHECK_ACL(aclrtMemcpy(xDevice, inputByteSize, xHost, inputByteSize, ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK_ACL(aclrtMemcpy(yDevice, inputByteSize, yHost, inputByteSize, ACL_MEMCPY_HOST_TO_DEVICE));
        // 用内核调用符<<<>>>调用核函数完成指定的运算,add_custom_do中封装了<<<>>>调用
        add_custom_do(numBlocks, nullptr, stream, xDevice, yDevice, zDevice);
        // 用ACLRT_LAUNCH_KERNEL接口调用核函数完成指定的运算
        // ACLRT_LAUNCH_KERNEL(add_custom)(numBlocks, stream, xDevice, yDevice, zDevice);
        CHECK_ACL(aclrtSynchronizeStream(stream));
        // 将Device上的运算结果拷贝回Host
        CHECK_ACL(aclrtMemcpy(zHost, outputByteSize, zDevice, outputByteSize, ACL_MEMCPY_DEVICE_TO_HOST));
        WriteFile("./output/output_z.bin", zHost, outputByteSize);
        // 释放申请的资源
        CHECK_ACL(aclrtFree(xDevice));
        CHECK_ACL(aclrtFree(yDevice));
        CHECK_ACL(aclrtFree(zDevice));
        CHECK_ACL(aclrtFreeHost(xHost));
        CHECK_ACL(aclrtFreeHost(yHost));
        CHECK_ACL(aclrtFreeHost(zHost));
        // 去初始化
        CHECK_ACL(aclrtDestroyStream(stream));
        CHECK_ACL(aclrtResetDevice(deviceId));
        CHECK_ACL(aclFinalize());
    ```

    > **说明：** 
    >针对<<<\>\>\>调用方式在[核函数](核函数.md)章节已有说明，这里仅对ACLRT\_LAUNCH\_KERNEL调用接口的使用方法介绍如下：
    >```
    >ACLRT_LAUNCH_KERNEL(kernel_name)(numBlocks, stream, argument list);
    >```
    >-   kernel\_name：算子核函数的名称。
    >-   numBlocks：规定了核函数将会在几个核上执行。每个执行该核函数的核会被分配一个逻辑ID，即block\_idx，可以在核函数的实现中调用[GetBlockIdx](GetBlockIdx.md)来获取block\_idx。
    >-   stream，类型为aclrtStream，stream用于维护一些异步操作的执行顺序，确保按照应用程序中的代码调用顺序在Device上执行。stream创建等管理接口请参考《应用开发指南 \(C&C++\)》中的“acl API参考 \> 运行时管理 \> Stream管理”章节。
    >-   argument list：参数列表，与核函数的参数列表保持一致。

## CMake编译配置文件编写<a name="section185111259496"></a>

本节会介绍CMake文件中一些关键环境变量和Cmake命令参数的说明，通常情况下不需要开发者修改，但是这些参数可以帮助开发者更好的理解编译原理，方便有能力的开发者对Cmake进行定制化处理。

**表 1**  环境变量说明

<a name="table8269104175818"></a>
<table><thead align="left"><tr id="row126924116583"><th class="cellrowborder" valign="top" width="24.54%" id="mcps1.2.3.1.1"><p id="p182692415584"><a name="p182692415584"></a><a name="p182692415584"></a>环境变量</p>
</th>
<th class="cellrowborder" valign="top" width="75.46000000000001%" id="mcps1.2.3.1.2"><p id="p12269124113588"><a name="p12269124113588"></a><a name="p12269124113588"></a>配置说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row13269204175816"><td class="cellrowborder" valign="top" width="24.54%" headers="mcps1.2.3.1.1 "><p id="p4269184195817"><a name="p4269184195817"></a><a name="p4269184195817"></a>SOC_VERSION</p>
</td>
<td class="cellrowborder" valign="top" width="75.46000000000001%" headers="mcps1.2.3.1.2 "><p id="p11449135425916"><a name="p11449135425916"></a><a name="p11449135425916"></a>AI处理器的型号。</p>
</td>
</tr>
<tr id="row102691419587"><td class="cellrowborder" valign="top" width="24.54%" headers="mcps1.2.3.1.1 "><p id="p4269134125810"><a name="p4269134125810"></a><a name="p4269134125810"></a>ASCEND_CANN_PACKAGE_PATH</p>
</td>
<td class="cellrowborder" valign="top" width="75.46000000000001%" headers="mcps1.2.3.1.2 "><p id="p7276191714015"><a name="p7276191714015"></a><a name="p7276191714015"></a>CANN软件包安装后的实际路径。</p>
</td>
</tr>
<tr id="row926994114584"><td class="cellrowborder" valign="top" width="24.54%" headers="mcps1.2.3.1.1 "><p id="p17269154195814"><a name="p17269154195814"></a><a name="p17269154195814"></a>CMAKE_BUILD_TYPE</p>
</td>
<td class="cellrowborder" valign="top" width="75.46000000000001%" headers="mcps1.2.3.1.2 "><p id="p623215581058"><a name="p623215581058"></a><a name="p623215581058"></a>编译模式选项，可配置为：</p>
<a name="ul91941346191017"></a><a name="ul91941346191017"></a><ul id="ul91941346191017"><li>“Release”，Release版本，不包含调试信息，编译最终发布的版本。</li><li>“Debug”，Debug版本，包含调试信息，便于开发者开发和调试。</li></ul>
</td>
</tr>
<tr id="row12269154116585"><td class="cellrowborder" valign="top" width="24.54%" headers="mcps1.2.3.1.1 "><p id="p202699414583"><a name="p202699414583"></a><a name="p202699414583"></a>CMAKE_INSTALL_PREFIX</p>
</td>
<td class="cellrowborder" valign="top" width="75.46000000000001%" headers="mcps1.2.3.1.2 "><p id="p633110348315"><a name="p633110348315"></a><a name="p633110348315"></a>用于指定CMAKE执行install时，安装的路径前缀，执行install后编译产物（ascendc_library中指定的target以及对应的头文件）会安装在该路径下。默认路径为当前目录的out目录下。</p>
</td>
</tr>
<tr id="row118991832132718"><td class="cellrowborder" valign="top" width="24.54%" headers="mcps1.2.3.1.1 "><p id="p68991732182716"><a name="p68991732182716"></a><a name="p68991732182716"></a>CMAKE_CXX_COMPILER_LAUNCHER</p>
</td>
<td class="cellrowborder" valign="top" width="75.46000000000001%" headers="mcps1.2.3.1.2 "><p id="p163551740194620"><a name="p163551740194620"></a><a name="p163551740194620"></a>用于配置C++语言编译器（如g++）、毕昇编译器的启动器程序为ccache，配置后即可开启cache缓存编译，<span>加速重复编译并提高构建效率</span>。使用该功能前需要安装ccache。</p>
<p id="p1182316525520"><a name="p1182316525520"></a><a name="p1182316525520"></a>配置方法如下，在对应的CMakeLists.txt进行设置：</p>
<pre class="screen" id="screen1421364716596"><a name="screen1421364716596"></a><a name="screen1421364716596"></a>set(CMAKE_CXX_COMPILER_LAUNCHER &lt;launcher_program&gt;)</pre>
<p id="p1235210514299"><a name="p1235210514299"></a><a name="p1235210514299"></a>其中&lt;launcher_program&gt;是ccache的安装路径，比如ccache的安装路径为/usr/bin/ccache，示例如下：</p>
<pre class="screen" id="screen554167313"><a name="screen554167313"></a><a name="screen554167313"></a>set(CMAKE_CXX_COMPILER_LAUNCHER /usr/bin/ccache)</pre>
</td>
</tr>
</tbody>
</table>

**表 2**  Cmake命令语法说明

<a name="table481718169817"></a>
<table><thead align="left"><tr id="row1981751617812"><th class="cellrowborder" valign="top" width="24.67%" id="mcps1.2.3.1.1"><p id="p188171016288"><a name="p188171016288"></a><a name="p188171016288"></a>Cmake命令</p>
</th>
<th class="cellrowborder" valign="top" width="75.33%" id="mcps1.2.3.1.2"><p id="p481751615812"><a name="p481751615812"></a><a name="p481751615812"></a>语法说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row1481717161086"><td class="cellrowborder" valign="top" width="24.67%" headers="mcps1.2.3.1.1 "><p id="p08176161484"><a name="p08176161484"></a><a name="p08176161484"></a>add_executable</p>
</td>
<td class="cellrowborder" valign="top" width="75.33%" headers="mcps1.2.3.1.2 "><p id="p15361746592"><a name="p15361746592"></a><a name="p15361746592"></a>使用指定的源文件将可执行文件添加到项目中。和Cmake通用的命令参数使用方法一致。</p>
</td>
</tr>
<tr id="row158188167811"><td class="cellrowborder" valign="top" width="24.67%" headers="mcps1.2.3.1.1 "><p id="p151562145914"><a name="p151562145914"></a><a name="p151562145914"></a>ascendc_library</p>
</td>
<td class="cellrowborder" valign="top" width="75.33%" headers="mcps1.2.3.1.2 "><p id="p83571846498"><a name="p83571846498"></a><a name="p83571846498"></a>使用指定的核函数源文件向项目（project）添加库。语法格式如下：</p>
<pre class="screen" id="screen894213141527"><a name="screen894213141527"></a><a name="screen894213141527"></a>ascendc_library(&lt;target_name&gt; [STATIC | SHARED]
            [&lt;source&gt;...]) </pre>
<p id="p211019713522"><a name="p211019713522"></a><a name="p211019713522"></a>其中&lt;target_name&gt;表示库文件的名字，该库文件会根据命令里列出的源文件来建立。STATIC、SHARED的作用是指定生成的库文件的类型。STATIC库是目标文件的归档文件，在连接其它目标的时候使用。SHARED库会被动态连接（动态连接库），在运行时会被加载。&lt;source&gt;表示核函数源文件。</p>
</td>
</tr>
<tr id="row7914182912215"><td class="cellrowborder" valign="top" width="24.67%" headers="mcps1.2.3.1.1 "><p id="p11914929132219"><a name="p11914929132219"></a><a name="p11914929132219"></a>ascendc_fatbin_library</p>
</td>
<td class="cellrowborder" valign="top" width="75.33%" headers="mcps1.2.3.1.2 "><p id="p1887436151919"><a name="p1887436151919"></a><a name="p1887436151919"></a>使用指定的核函数源文件编译生成一个Kernel二进制文件，供Kernel加载和执行接口使用。语法格式如下：</p>
<pre class="screen" id="screen487173618193"><a name="screen487173618193"></a><a name="screen487173618193"></a>ascendc_fatbin_library(&lt;target_name&gt; [&lt;source&gt;...]) </pre>
<a name="ul191554253714"></a><a name="ul191554253714"></a><ul id="ul191554253714"><li>&lt;target_name&gt;表示库文件的名字，该库文件会根据命令里列出的核函数源文件编译生成&lt;target_name&gt;.o文件，放置于${CMAKE_INSTALL_PREFIX}/fatbin/${target_name}/路径下。</li><li>&lt;source&gt;表示核函数源文件。</li></ul>
<div class="note" id="note96252591853"><a name="note96252591853"></a><a name="note96252591853"></a><span class="notetitle"> 说明： </span><div class="notebody"><a name="ul143550276817"></a><a name="ul143550276817"></a><ul id="ul143550276817"><li>Kernel加载与执行接口的调用流程和上文介绍的&lt;&lt;&lt;...&gt;&gt;&gt;等调用流程有所差异，具体流程请参考<span id="ph12767132811535"><a name="ph12767132811535"></a><a name="ph12767132811535"></a>《应用开发指南 (C&amp;C++)》</span>中的<span id="ph625615172587"><a name="ph625615172587"></a><a name="ph625615172587"></a>“acl API参考 &gt; 运行时管理 &gt; Kernel加载与执行”</span>章节。</li><li>该编译选项暂不支持printf、DumpTensor、DumpAccChkPoint、assert接口。</li></ul>
</div></div>
</td>
</tr>
<tr id="row681816162814"><td class="cellrowborder" valign="top" width="24.67%" headers="mcps1.2.3.1.1 "><p id="p1481814164815"><a name="p1481814164815"></a><a name="p1481814164815"></a>ascendc_compile_definitions</p>
</td>
<td class="cellrowborder" valign="top" width="75.33%" headers="mcps1.2.3.1.2 "><p id="p1344313389320"><a name="p1344313389320"></a><a name="p1344313389320"></a>添加编译宏。可以添加<span id="zh-cn_topic_0000001552186366_ph35921824172811"><a name="zh-cn_topic_0000001552186366_ph35921824172811"></a><a name="zh-cn_topic_0000001552186366_ph35921824172811"></a>Ascend C</span>提供的编译宏和开发者自定义的编译宏。语法格式如下：</p>
<pre class="screen" id="screen5975184113338"><a name="screen5975184113338"></a><a name="screen5975184113338"></a>ascendc_compile_definitions(&lt;target_name&gt; [PRIVATE]
            [&lt;xxx&gt;...]) </pre>
<p id="p1655714242611"><a name="p1655714242611"></a><a name="p1655714242611"></a><span id="ph12925144843119"><a name="ph12925144843119"></a><a name="ph12925144843119"></a>Ascend C</span>提供的编译宏介绍如下：</p>
<a name="ul95991737102913"></a><a name="ul95991737102913"></a><ul id="ul95991737102913"><li>HAVE_WORKSPACE用于表示kernel入口是否包含workspace入参。默认情况下为不包含；增加该编译宏后，表示包含，此时框架会获取kernel入参的倒数第一个参数（未配置<a href="#li6933155615394">HAVE_TILING</a>），或倒数第二个参数（配置HAVE_TILING），自动在kernel侧设置系统workspace，开发者在kernel侧入参处获取的workspace为偏移了系统workspace后的用户workspace。当开发者使用了<a href="Matmul-Kernel侧接口.md">Matmul Kernel侧接口</a>等需要系统workspace的高阶API时，建议开启此参数，入参排布、系统workspace的设置逻辑与<a href="工程化算子开发.md">工程化算子开发</a>保持一致，可减少算子实现在不同开发方式间切换带来的修改成本。需要注意的是，host侧开发者仍需要自行申请workspace的空间，系统workspace大小可以通过<a href="PlatformAscendCManager.md">PlatformAscendCManager</a>的<a href="GetLibApiWorkSpaceSize.md">GetLibApiWorkSpaceSize</a>接口获取。HAVE_WORKSPACE的设置样例如下：<pre class="screen" id="screen170874019451"><a name="screen170874019451"></a><a name="screen170874019451"></a>ascendc_compile_definitions(ascendc_kernels_${RUN_MODE} PRIVATE
    HAVE_WORKSPACE
)</pre>
</li><li id="li6933155615394"><a name="li6933155615394"></a><a name="li6933155615394"></a>HAVE_TILING用于表示kernel入口是否含有tiling入参。在配置了HAVE_WORKSPACE之后，此编译宏才会生效。默认情况下为不包含，开关关闭；增加该编译宏后，表示包含，此时框架会将kernel入参的最后一个参数当做tiling，将倒数第二个参数当做workspace。框架不会对此tiling入参做任何处理，仅通过该入参来判断workspace参数的位置，使用此编译宏可以和<a href="工程化算子开发.md">工程化算子开发</a>保持入参一致，减少算子实现在不同开发方式间切换带来的修改成本。设置样例如下：<pre class="screen" id="screen8986115020458"><a name="screen8986115020458"></a><a name="screen8986115020458"></a>ascendc_compile_definitions(ascendc_kernels_${RUN_MODE} PRIVATE
    HAVE_WORKSPACE
    HAVE_TILING
)</pre>
</li></ul>
</td>
</tr>
<tr id="row66361192115"><td class="cellrowborder" valign="top" width="24.67%" headers="mcps1.2.3.1.1 "><p id="p2105162111117"><a name="p2105162111117"></a><a name="p2105162111117"></a>ascendc_compile_options</p>
</td>
<td class="cellrowborder" valign="top" width="75.33%" headers="mcps1.2.3.1.2 "><p id="p54471936201110"><a name="p54471936201110"></a><a name="p54471936201110"></a>添加编译选项。可以添加相应的编译选项用于host侧与device侧的编译过程。语法格式如下：</p>
<pre class="screen" id="screen96578372136"><a name="screen96578372136"></a><a name="screen96578372136"></a>ascendc_compile_options(&lt;target_name&gt; PRIVATE
    [&lt;xxx&gt;...]
)</pre>
<p id="p4971111411712"><a name="p4971111411712"></a><a name="p4971111411712"></a>默认情况下，指定的编译选项都将传递给device侧编译器进行编译。若想传递编译选项给host侧编译器，则需要使用“-forward-options-to-host-compiler”编译选项，该选项后的编译选项将传递给host侧编译器，示例如下：</p>
<pre class="screen" id="screen177651655121717"><a name="screen177651655121717"></a><a name="screen177651655121717"></a>ascendc_compile_options(&lt;target_name&gt; PRIVATE
    -g
    -forward-options-to-host-compiler
    -gdwarf-4
)</pre>
<p id="p124791133141310"><a name="p124791133141310"></a><a name="p124791133141310"></a>如上代码所示，在编译时，“-g”编译选项传递给device侧编译器，“-gdwarf-4”编译选项传递给host侧编译器。</p>
<p id="p12628533191311"><a name="p12628533191311"></a><a name="p12628533191311"></a>备注：host侧编译选项只支持g++与clang编译器共同支持的编译选项。</p>
</td>
</tr>
<tr id="row13716256162616"><td class="cellrowborder" valign="top" width="24.67%" headers="mcps1.2.3.1.1 "><p id="p371719565262"><a name="p371719565262"></a><a name="p371719565262"></a>ascendc_include_directories</p>
</td>
<td class="cellrowborder" valign="top" width="75.33%" headers="mcps1.2.3.1.2 "><p id="p153783519322"><a name="p153783519322"></a><a name="p153783519322"></a>添加开发者自定义的头文件搜索路径。语法格式如下：</p>
<pre class="screen" id="screen2379195143211"><a name="screen2379195143211"></a><a name="screen2379195143211"></a>ascendc_include_directories(&lt;target_name&gt; [PRIVATE]
            [&lt;xxx&gt;...]) </pre>
</td>
</tr>
</tbody>
</table>

简化的编译流程图如下图所示：将算子核函数源文件编译生成kernel侧的库文件（\*.so或\*.a库文件）；工程框架自动生成核函数调用接口声明头文件；编译main.cpp（算子调用应用程序）时依赖上述头文件，将编译应用程序生成的目标文件和kernel侧的库文件进行链接，生成最终的可执行文件。

**图 3**  编译简化流程图<a name="fig744344916358"></a>  
<!-- img2text -->
```
out

┌───────────────────────────────────┐
│ add_custom.cpp                    │
│ matmul_custom.cpp                 │
│ ...                               │
└───────────────────────────────────┘
                 │
                 │
            ┌─────────┐
            │ Compile │
            └─────────┘
                 │
                 ▼
      ┌──────────────────────┐
      │ libkernels1.a        │
      │ libkernels2.so       │
      │ ...                  │
      └──────────────────────┘
                 │
                 │
                 │
                 │
┌───────────────────────────────────┐
│ 自动生成的                         │
│ aclrtlaunch_add_custom.h          │
│ aclrtlaunch_matmul_custom.h       │
│ ...                               │
└───────────────────────────────────┘
                 │
                 ├──────────────────────┐
                 │                      │
                 │                      │
┌───────────────────────────────────┐  │
│ main.cpp                          │  │
└───────────────────────────────────┘  │
                 │                      │
                 │                      │
            ┌─────────┐                │
            │ Compile │                │
            └─────────┘                │
                 │                      │
                 ▼                      │
      ┌──────────────────────┐          │
      │ 目标文件             │◀─────────┘
      └──────────────────────┘
                 │
                 │
                 ├──────────────────────┐
                 │                      │
                 │                      │
            ┌────┴────┐                 │
            │  Link   │◀────────────────┘
            └─────────┘
                 │
                 ▼
      ┌──────────────────────┐
      │ 可执行文件           │
      │ main                 │
      └──────────────────────┘
```

编译安装结束后在CMAKE\_INSTALL\_PREFIX目录下生成的编译产物示例如下；最终的可执行文件会生成在cmake命令的执行目录下。

```
out
├── lib 
│   ├── libkernels1.a
│   ├── libkernels2.so
├── include
│   ├── kernels1
│           ├── aclrtlaunch_matmul_custom.h
│           ├── aclrtlaunch_add_custom.h
│   ├── kernels2
│           ├── aclrtlaunch_xxx.h
│           ├── ...
```

对于lib目录下生成的库文件可通过msobjdump工具进一步解析得到kernel信息，具体操作参见[msobjdump工具](msobjdump工具.md)。

## 输入数据和真值数据生成以及验证脚本文件<a name="section1234873541816"></a>

以固定shape的add\_custom算子为例，输入数据和真值数据生成的脚本样例如下：根据算子的输入输出编写脚本，生成输入数据和真值数据。

```
#!/usr/bin/python3
# -*- coding:utf-8 -*-
import numpy as np

def gen_golden_data_simple():
    input_x = np.random.uniform(1, 100, [8, 2048]).astype(np.float16)
    input_y = np.random.uniform(1, 100, [8, 2048]).astype(np.float16)
    golden = (input_x + input_y).astype(np.float16)

    input_x.tofile("./input/input_x.bin")
    input_y.tofile("./input/input_y.bin")
    golden.tofile("./output/golden.bin")

if __name__ == "__main__":
    gen_golden_data_simple()
```

验证输出数据和真值数据是否一致的验证脚本样例如下：当前使用numpy接口计算了输出数据和真值数据的绝对误差和相对误差，误差在容忍偏差范围内，视为精度符合要求，输出"test pass"字样。

```
import os
import sys
import numpy as np

loss = 1e-3 # 容忍偏差，一般fp16要求绝对误差和相对误差均不超过千分之一
minimum = 10e-10

def verify_result(real_result, golden):
    real_result = np.fromfile(real_result, dtype=np.float16) # 从bin文件读取实际运算结果
    golden = np.fromfile(golden, dtype=np.float16) # 从bin文件读取预期运算结果
    result = np.abs(real_result - golden) # 计算运算结果和预期结果偏差
    deno = np.maximum(np.abs(real_result), np.abs(golden)) # 获取最大值并组成新数组
    result_atol = np.less_equal(result, loss) # 计算绝对误差
    result_rtol = np.less_equal(result / np.add(deno, minimum), loss) # 计算相对误差
    if not result_rtol.all() and not result_atol.all():
        if np.sum(result_rtol == False) > real_result.size * loss and np.sum(result_atol == False) > real_result.size * loss:
            print("[ERROR] result error")
            return False
    print("test pass")
    return True

if __name__ == '__main__':
    verify_result(sys.argv[1],sys.argv[2])
```

## 修改并执行一键式编译运行脚本<a name="section188001652105215"></a>

您可以基于样例工程中提供的一键式编译运行脚本进行快速编译，并在CPU侧和NPU侧执行Ascend C算子。一键式编译运行脚本主要完成以下功能：

**图 4**  一键式编译运行脚本流程图<a name="fig125041443583"></a>  
<!-- img2text -->
```text
                ┌──────┐
                │ 开始 │
                └──┬───┘
                   │
                   ▼
┌──────────────────────────────────────┐      ┌──────────────────────────────────────────────┐
│ 输入数据和真值数据生成脚本           │ ───→ │ 样例工程input目录下                           │
└──────────────────┬───────────────────┘      │ 生成输入数据：input_x.bin、input_y.bin       │
                   │                          └──────────────────────────────────────────────┘
                   ├───────────────────────→  ┌──────────────────────────────────────────────┐
                   │                          │ 样例工程output目录下                          │
                   │                          │ 生成真值数据：golden.bin                      │
                   │                          └──────────────────────────────────────────────┘
                   ▼
┌──────────────────────────────────────┐      ┌──────────────────────────────────────────────┐
│ CMAKE编译算子，生成可执行文件        │ ───→ │ 样例工程out/bin/目录下                        │
└──────────────────┬───────────────────┘      │ 生成可执行文件：                              │
                   │                          │ ascendc_kernels_bbit                         │
                   │                          └──────────────────────────────────────────────┘
                   ▼
┌──────────────────────────────────────┐      ┌──────────────────────────────────────────────┐
│ 执行编译生成的可执行文件，           │ ───→ │ 样例工程output目录下                          │
│ 生成算子实际输出结果                 │      │ 生成实际输出结果：output_z.bin                │
└──────────────────┬───────────────────┘      └──────────────────────────┬───────────────────┘
                   │                                                     │
                   ▼                                                     │
┌──────────────────────────────────────┐                                 │
│ 使用numpy接口计算了输出数据          │ ─────────────────────────────→  ┌─────────┐
│ 和真值数据的绝对误差和相对误差，     │                                 │ Compare │
│ 误差在容忍偏差范围内，视             │                                 └────┬────┘
│ 为精度符合要求                       │                                      │
└──────────────────┬───────────────────┘                                      │
                   │                                                          │
                   ▼                                                          │
                ┌──────┐                                                      │
                │ 结束 │                                                      │
                └──────┘                                                      │
                                                                               │
                                                                               └────────────→
                                                                                 指向“生成真值数据：golden.bin”
```

> **须知：** 
>**样例中提供的一键式编译运行脚本并不能适用于所有的算子运行验证场景，使用时请根据实际情况进行修改。**
>-   根据Ascend C算子的算法原理的不同，自行实现输入和真值数据的生成脚本。

完成上述文件的编写后，可以执行一键式编译运行脚本，编译和运行应用程序。

脚本执行方式和脚本参数介绍如下：

```
bash run.sh --run-mode=npu  --soc-version=<soc_version> --install-path=<install_path>  --build-type=Debug  --install-prefix=<install-prefix>

bash run.sh -r npu  -v <soc_version> -i <install_path> -b Debug -p <install-prefix>
```

**表 3**  脚本参数介绍

<a name="table98393011180"></a>
<table><thead align="left"><tr id="row98396051814"><th class="cellrowborder" valign="top" width="16.14%" id="mcps1.2.4.1.1"><p id="p283916071814"><a name="p283916071814"></a><a name="p283916071814"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="11.690000000000001%" id="mcps1.2.4.1.2"><p id="p1872572854511"><a name="p1872572854511"></a><a name="p1872572854511"></a>参数简写</p>
</th>
<th class="cellrowborder" valign="top" width="72.17%" id="mcps1.2.4.1.3"><p id="p683914011181"><a name="p683914011181"></a><a name="p683914011181"></a>参数介绍</p>
</th>
</tr>
</thead>
<tbody><tr id="row1894192514199"><td class="cellrowborder" valign="top" width="16.14%" headers="mcps1.2.4.1.1 "><p id="p1552703816245"><a name="p1552703816245"></a><a name="p1552703816245"></a>--run-mode</p>
</td>
<td class="cellrowborder" valign="top" width="11.690000000000001%" headers="mcps1.2.4.1.2 "><p id="p19726172814452"><a name="p19726172814452"></a><a name="p19726172814452"></a>-r</p>
</td>
<td class="cellrowborder" valign="top" width="72.17%" headers="mcps1.2.4.1.3 "><p id="p152753813245"><a name="p152753813245"></a><a name="p152753813245"></a>表明算子以cpu模式或npu模式运行。</p>
<p id="p9527838172410"><a name="p9527838172410"></a><a name="p9527838172410"></a>取值为cpu或npu。</p>
</td>
</tr>
<tr id="row12894625191917"><td class="cellrowborder" valign="top" width="16.14%" headers="mcps1.2.4.1.1 "><p id="p198251262416"><a name="p198251262416"></a><a name="p198251262416"></a>--soc-version</p>
</td>
<td class="cellrowborder" valign="top" width="11.690000000000001%" headers="mcps1.2.4.1.2 "><p id="p572682874519"><a name="p572682874519"></a><a name="p572682874519"></a>-v</p>
</td>
<td class="cellrowborder" valign="top" width="72.17%" headers="mcps1.2.4.1.3 "><p id="p1931033151815"><a name="p1931033151815"></a><a name="p1931033151815"></a>算子运行的AI处理器型号。</p>
<div class="note" id="note15861181414615"><a name="note15861181414615"></a><a name="note15861181414615"></a><span class="notetitle"> 说明： </span><div class="notebody"></div></div>
</td>
</tr>
<tr id="row789432514198"><td class="cellrowborder" valign="top" width="16.14%" headers="mcps1.2.4.1.1 "><p id="p1960112537220"><a name="p1960112537220"></a><a name="p1960112537220"></a>--install-path</p>
</td>
<td class="cellrowborder" valign="top" width="11.690000000000001%" headers="mcps1.2.4.1.2 "><p id="p8726428124511"><a name="p8726428124511"></a><a name="p8726428124511"></a>-i</p>
</td>
<td class="cellrowborder" valign="top" width="72.17%" headers="mcps1.2.4.1.3 "><p id="p2092216317349"><a name="p2092216317349"></a><a name="p2092216317349"></a>配置为CANN软件的安装路径，请根据实际安装路径进行修改。</p>
<p id="p92878327253"><a name="p92878327253"></a><a name="p92878327253"></a>默认值为$HOME/Ascend/ascend-toolkit/latest。</p>
</td>
</tr>
<tr id="row4894725151916"><td class="cellrowborder" valign="top" width="16.14%" headers="mcps1.2.4.1.1 "><p id="p04911128122019"><a name="p04911128122019"></a><a name="p04911128122019"></a>--build-type</p>
</td>
<td class="cellrowborder" valign="top" width="11.690000000000001%" headers="mcps1.2.4.1.2 "><p id="p972682894515"><a name="p972682894515"></a><a name="p972682894515"></a>-b</p>
</td>
<td class="cellrowborder" valign="top" width="72.17%" headers="mcps1.2.4.1.3 "><p id="p1368918227265"><a name="p1368918227265"></a><a name="p1368918227265"></a>编译模式选项，可配置为：</p>
<a name="ul136897222265"></a><a name="ul136897222265"></a><ul id="ul136897222265"><li>Release，Release版本，不包含调试信息，编译最终发布的版本。</li><li>Debug，Debug版本，包含调试信息，便于开发者开发和调试。</li></ul>
<p id="p298174118283"><a name="p298174118283"></a><a name="p298174118283"></a>默认值为Debug。</p>
</td>
</tr>
<tr id="row28943258198"><td class="cellrowborder" valign="top" width="16.14%" headers="mcps1.2.4.1.1 "><p id="p99268149290"><a name="p99268149290"></a><a name="p99268149290"></a>--install-prefix</p>
</td>
<td class="cellrowborder" valign="top" width="11.690000000000001%" headers="mcps1.2.4.1.2 "><p id="p9726122854513"><a name="p9726122854513"></a><a name="p9726122854513"></a>-p</p>
</td>
<td class="cellrowborder" valign="top" width="72.17%" headers="mcps1.2.4.1.3 "><p id="p43731433122017"><a name="p43731433122017"></a><a name="p43731433122017"></a>用于指定CMAKE执行install时，安装的路径前缀，执行install后编译产物（ascendc_library中指定的target以及对应的头文件）会安装在该路径下。默认路径为当前目录的out目录下。</p>
</td>
</tr>
</tbody>
</table>

脚本执行完毕输出"test pass"字样表示算子精度符合要求。

