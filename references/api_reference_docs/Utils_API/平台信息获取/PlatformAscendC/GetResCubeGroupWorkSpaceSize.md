# GetResCubeGroupWorkSpaceSize

**页面ID:** atlasascendc_api_07_1038  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_1038.html

---

#### 功能说明

基于CreateCubeResGroup进行AI Core分组计算需要传入workspace用于消息通信，在Host侧提供本接口用于获取CreateCubeResGroup所需要的workspace空间大小。

#### 函数原型

```
uint32_t GetResCubeGroupWorkSpaceSize(void) const
```

#### 参数说明

无

#### 返回值说明

当前CreateCubeResGroup所需要的workspace空间大小。

#### 约束说明

无。

#### 调用示例

```
// 用户自定义的tiling函数
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    AddApiTiling tiling;
    ...
    // 如需要使用系统workspace需要调用GetLibApiWorkSpaceSize获取系统workspace的大小。
    auto ascendcPlatform = platform_ascendc:: PlatformAscendC(context->GetPlatformInfo());
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    // 设置用户需要使用的workspace和CreateCubeResGroup需要的大小作为usrWorkspace的总大小。
    size_t usrSize = 256 + ascendcPlatform.GetResCubeGroupWorkSpaceSize(); 
    size_t *currentWorkspace = context->GetWorkspaceSizes(1); // 通过框架获取workspace的指针，GetWorkspaceSizes入参为所需workspace的块数。当前限制使用一块。
    currentWorkspace[0] = usrSize + sysWorkspaceSize; // 设置总的workspace的数值大小，总的workspace空间由框架来申请并管理。
    ...
}
```
