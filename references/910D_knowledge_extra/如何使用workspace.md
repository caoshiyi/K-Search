# 如何使用workspace<a name="ZH-CN_TOPIC_0000002554351537"></a>

workspace是设备侧Global Memory上的一块内存。workspace内存分为两部分：系统workspace和用户workspace。

-   系统workspace：Ascend C API需要预留的workspace内存

    API在计算过程需要一些workspace内存作为缓存，因此算子需要为API预留workspace内存，预留内存大小通过[GetLibApiWorkSpaceSize](GetLibApiWorkSpaceSize.md)接口获取。

-   用户workspace：算子实现使用到的workspace内存

    算子内部需要通过额外的device内存进行数据交换或者缓存的时候才需要分配，根据实际情况自行分配。使用场景如下：

    -   需要使用Unified Buffer和L1 Buffer上的空间且空间不够用时，可以将数据暂存至workspace上。
    -   调用[SyncAll](SyncAll.md)等API接口时，需要workspace作为入参。
    -   其他需要使用Global Memory上内存空间的场景。

不同开发方式下，具体的使用方法如下：

-   工程化算子开发方式

    在tiling函数中先通过GetWorkspaceSizes接口获取workspace大小的存放位置，再设置workspace的大小，框架侧会为其申请对应大小的设备侧Global Memory，在对应的算子kernel侧实现时可以使用这块workspace内存。在使用[Matmul Kernel侧接口](Matmul-Kernel侧接口.md)等需要系统workspace的高阶API时，设置的workspace空间大小为系统workspace和用户workspace之和。

    ```
    // 用户自定义的tiling函数
    static ge::graphStatus TilingFunc(gert::TilingContext* context)
    {
        AddApiTiling tiling;
        ...
        size_t usrSize = 256; // 设置用户需要使用的workspace大小为256字节。
        // 如需要使用系统workspace需要调用GetLibApiWorkSpaceSize获取系统workspace的大小。
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
        size_t *currentWorkspace = context->GetWorkspaceSizes(1); // 通过框架获取workspace的指针，GetWorkspaceSizes入参为所需workspace的块数。当前限制使用一块。
        currentWorkspace[0] = usrSize + sysWorkspaceSize; // 设置总的workspace的数值大小，总的workspace空间由框架来申请并管理。
        ...
    }
    ```

    在device侧kernel入口处的workspace为用户的workspace指针：

    ```
    // 用户写的Kernel函数，核函数必须包括GM_ADDR workspace入参，位置需要放在tiling之前
    extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
    {
        ...
        
    }
    ```

-   Kernel直调算子开发场景

    需要使用workspace空间时，建议开启编译选项[HAVE\_WORKSPACE](基于样例工程完成Kernel直调.md#li1103101565014)。host侧开发者仍需要自行申请workspace的空间，并传入。在使用[Matmul Kernel侧接口](Matmul-Kernel侧接口.md)等需要系统workspace的高阶API时，设置的workspace空间大小为系统workspace和用户workspace之和。系统workspace大小可以通过[PlatformAscendCManager](PlatformAscendCManager.md)的GetLibApiWorkSpaceSize接口获取。开启[HAVE\_WORKSPACE](基于样例工程完成Kernel直调.md#li1103101565014)后，开发者在kernel侧入参处获取的workspace为偏移了系统workspace后的用户workspace。

