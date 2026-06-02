# 如何进行Tiling调测

**页面ID:** atlas_ascendc_10_00018  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_10_00018.html

---

在工程化算子开发过程中，开发者需实现Tiling函数，该函数原型是固定的，接受TilingContext作为输入。框架负责构造TilingContext并调用Tiling函数。若需单独进行Tiling调测，开发者可通过OpTilingRegistry加载编译后的Tiling动态库，获取Tiling函数的指针并进行调用，调用时Tiling函数的TilingContext入参使用ContextBuilder构建。

以下是具体步骤：

1. 参考工程化算子开发的开发步骤，完成算子实现，并通过**算子包编译**或**算子动态库编译**获取对应的Tiling动态库文件。

  - 算子包编译：Tiling实现对应的动态库为算子包部署目录下的liboptiling.so。具体路径可参考算子包部署。
  - 动态库编译：Tiling实现集成在算子动态库libcust_opapi.so中。具体路径可参考算子动态库和静态库编译。

2. 编写测试代码。

  - 使用ContextBuilder配置输入输出Tensor的形状、数据类型、格式及平台信息等，构建TilingContext。
  - 通过OpTilingRegistry的LoadTilingLibrary接口加载Tiling动态库；使用GetTilingFunc接口获取Tiling函数指针。
  - 执行Tiling函数，验证其正确性。

```
// test.cpp
#include <iostream>
#include "exe_graph/runtime/storage_shape.h"
#include "tiling/context/context_builder.h"

int main()
{
    gert::StorageShape x_shape = {{2, 32}, {2, 32}};
    gert::StorageShape y_shape = {{2, 32}, {2, 32}};
    gert::StorageShape z_shape = {{2, 32}, {2, 32}};

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holder.get());

    auto holder = context_ascendc::ContextBuilder()
                                .NodeIoNum(2, 1)
                                .IrInstanceNum({1, 1})
                                .AddInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND, x_shape)
                                .AddInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND, y_shape)
                                .AddOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND, z_shape)
                                .TilingData(param.get())
                                .Workspace(ws_size)
                                .AddPlatformInfo("Ascendxxxyy")
                                .BuildTilingContext();
    auto tilingContext = holder->GetContext<gert::TilingContext>();
    context_ascendc::OpTilingRegistry tmpIns;
    bool flag = tmpIns.LoadTilingLibrary("/your/path/to/so_path/liboptiling.so");  // 加载对应的Tiling动态库文件
    if (flag == false) {
        std::cout << "Failed to load tiling so" << std::endl;
        return -1;
    }
    context_ascendc::TilingFunc tilingFunc = tmpIns.GetTilingFunc("AddCustom");  // 获取AddCustom算子对应的Tiling函数, 此处入参为OpType
    if (tilingFunc != nullptr) {
        ge::graphStatus ret = tilingFunc(tilingContext);  // 执行Tiling函数
        if (ret != ge::GRAPH_SUCCESS) {
            std::cout << "Exec tiling func failed." << std::endl;
            return -1;
        }
    } else {
        std::cout << "Get tiling func failed." << std::endl;
        return -1;
    }
    return 0;
}
```

3. 编译测试代码。

```
g++ test.cpp -I${INSTALL_DIR}/include  -L${INSTALL_DIR}/lib64 -Wl,-rpath,${INSTALL_DIR}/lib64 -ltiling_api -lc_sec -lgraph_base -lregister -lascendalog -lplatform -o test
```

  - ${INSTALL_DIR}请替换为CANN软件安装后文件存储路径。以root用户安装为例，则安装后文件存储路径为：/usr/local/Ascend/cann。
  - 开发者根据需要链接依赖的动态库，必需链接的动态库有：

    - libtiling_api.so：Tiling功能相关的动态库，包含ContextBuilder类、OpTilingRegistry类等。
    - libc_sec.so：安全函数库，libtiling_api.so依赖该库。
    - libgraph_base.so：基础数据结构与接口库，libtiling_api.so依赖该库。
    - libregister.so：业务函数注册相关库（例如Tiling函数注册，算子原型注册等）。
    - libascendalog.so：log库，libtiling_api.so依赖该库。
    - libplatform.so：平台信息库，libtiling_api.so依赖该库；Tiling函数中使用硬件平台信息时，需要依赖该库。

4. 执行可执行文件。

```
./test
```
