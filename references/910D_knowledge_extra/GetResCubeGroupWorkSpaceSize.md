# GetResCubeGroupWorkSpaceSize<a name="ZH-CN_TOPIC_0000002554343681"></a>

## 功能说明<a name="zh-cn_topic_0000001969965338_zh-cn_topic_0000001391767420_section36583473819"></a>

基于[CreateCubeResGroup](CreateCubeResGroup.md)进行AI Core分组计算需要传入workspace用于消息通信，在Host侧提供本接口用于获取CreateCubeResGroup所需要的workspace空间大小。

## 函数原型<a name="zh-cn_topic_0000001969965338_zh-cn_topic_0000001391767420_section13230182415108"></a>

```
uint32_t GetResCubeGroupWorkSpaceSize(void) const
```

## 参数说明<a name="zh-cn_topic_0000001969965338_zh-cn_topic_0000001391767420_section189014013619"></a>

无

## 返回值说明<a name="zh-cn_topic_0000001969965338_zh-cn_topic_0000001391767420_section25791320141317"></a>

当前CreateCubeResGroup所需要的workspace空间大小。

## 约束说明<a name="zh-cn_topic_0000001969965338_zh-cn_topic_0000001391767420_section19165124931511"></a>

无。

## 调用示例<a name="zh-cn_topic_0000001969965338_zh-cn_topic_0000001391767420_section320753512363"></a>

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

