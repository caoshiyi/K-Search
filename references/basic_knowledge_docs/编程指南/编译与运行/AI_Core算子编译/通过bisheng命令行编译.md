# 通过bisheng命令行编译

**页面ID:** atlas_ascendc_10_00037  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_10_00037.html

---

毕昇编译器是一款专为昇腾AI处理器设计的编译器，支持异构编程扩展，可以将用户编写的昇腾算子代码编译成二进制可执行文件和动态库等形式。毕昇编译器的可执行程序命名为bisheng，支持x86、aarch64等主机系统，并且原生支持设备侧AI Core架构指令集编译。通过使用毕昇编译器，用户可以更加高效地进行针对昇腾AI处理器的编程和开发工作。

#### 入门示例

以下是一个使用毕昇编译器编译静态Shape的add_custom算子入门示例。该示例展示了如何编写源文件add_custom.asc以及具体的编译命令。通过这个示例，您可以了解如何使用毕昇编译器进行算子编译。完整样例请参考[LINK](https://gitee.com/ascend/samples/tree/v1.9-8.3.RC1/operator/ascendc/0_introduction/25_simple_add)。

1. 包含头文件。

在编写算子源文件时，需要包含必要的头文件。

```
// 头文件
#include "acl/acl.h"
#include "kernel_operator.h"
```

2. 核函数实现。

  - 核函数支持模板。
  - 核函数入参支持传入用户自定义的结构体，比如示例中用户自定义的AddCustomTilingData结构体。

```
// 用户自定义的TilingData结构体
struct AddCustomTilingData { 
    uint32_t totalLength; 
    uint32_t tileNum;
};

// Kernel核心实现逻辑，包括搬运，计算等
class KernelAdd {
public:
    __aicore__ inline KernelAdd() {}
    // ...

};

__global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, AddCustomTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY); // 该算子执行时仅启动AI Core上的Vector核
    KernelAdd op;
    op.Init(x, y, z, tiling.totalLength, tiling.tileNum); 
    op.Process();
}
```

3. Host侧调用函数逻辑，包括内存申请和释放，初始化和去初始化，内核调用符调用核函数等。

```
// Host侧应用程序需要包含的头文件
#include "acl/acl.h"
// Kernel侧需要包含的头文件
#include "kernel_operator.h"
// 核函数开发部分
...

__global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, AddCustomTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);    
    KernelAdd op;
    op.Init(x, y, z, tiling.totalLength, tiling.tileNum);
    op.Process();
}

// 通过<<<...>>>内核调用符调用算子
std::vector<float> kernel_add(std::vector<float> &x, std::vector<float> &y)
{
...
}

// 计算结果比对
uint32_t VerifyResult(std::vector<float> &output, std::vector<float> &golden)
{
...
}

// 算子验证主程序
int32_t main(int32_t argc, char *argv[])
{
    constexpr uint32_t totalLength = 8 * 2048;
    constexpr float valueX = 1.2f;
    constexpr float valueY = 2.3f;
    std::vector<float> x(totalLength, valueX);
    std::vector<float> y(totalLength, valueY);

    std::vector<float> output = kernel_add(x, y);

    std::vector<float> golden(totalLength, valueX + valueY);
    return VerifyResult(output, golden);
}
```

4. 采用如下的编译命令进行编译。

  - -o demo：指定输出文件名为demo。
  - --npu-arch=dav-2201：指定NPU的架构版本为dav-2201。dav-后为NPU架构版本号，各产品型号对应的架构版本号请通过对应关系表进行查询。

```
bisheng add_custom.asc -o demo --npu-arch=dav-2201
```

5. 执行可执行文件。

```
./demo
```

#### 程序的编译与执行

通过毕昇编译器可以将算子源文件（以.asc为后缀）编译为当前平台的可执行文件或算子动态库，静态库。此外，也支持编译以.cpp/.c等为后缀的C++/C源文件，但需要增加-x asc编译选项。

- 编译生成可执行文件

```
# 1.编译hello_world.cpp为当前平台可执行文件 
# bisheng [算子源文件] -o [输出产物名称] --npu-arch=[NPU架构版本号]，常见参数顺序与g++保持一致。
bisheng -x asc add_custom.cpp -o add_custom --npu-arch=dav-xxxx
```

生成的可执行文件可通过如下方式执行：

```
./add_custom
```

- 编译生成算子动态库

```
# 2.编译add_custom_base.cpp生成算子动态库
# bisheng -shared [算子源文件] -o [输出产物名称] --npu-arch=[NPU架构版本号]
# 动态库
bisheng -shared -x asc add_custom_base.cpp -o libadd.so --npu-arch=dav-xxxx
```

- 编译生成算子静态库

```
# 3.编译add_custom_base.cpp生成算子静态库
bisheng -lib [算子源文件] -o [输出产物名称] --npu-arch=[NPU架构版本号]
# 静态库
bisheng -lib -x asc add_custom_base.cpp -o libadd.a --npu-arch=dav-xxxx
```

在命令行编译场景下，可以按需链接需要的库文件，常见的库文件请参考常用的链接库。编译时会默认链接表2中列出的库文件。注意如下例外场景：在使用g++链接asc代码编译生成的静态库时，需要手动链接默认链接库。
