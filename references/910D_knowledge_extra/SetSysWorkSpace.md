# SetSysWorkSpace<a name="ZH-CN_TOPIC_0000002523344174"></a>

## 产品支持情况<a name="section1550532418810"></a>

<a name="zh-cn_topic_0000002554424517_table38301303189"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000002554424517_row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0000002554424517_p1883113061818"><a name="zh-cn_topic_0000002554424517_p1883113061818"></a><a name="zh-cn_topic_0000002554424517_p1883113061818"></a><span id="zh-cn_topic_0000002554424517_ph20833205312295"><a name="zh-cn_topic_0000002554424517_ph20833205312295"></a><a name="zh-cn_topic_0000002554424517_ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="zh-cn_topic_0000002554424517_p783113012187"><a name="zh-cn_topic_0000002554424517_p783113012187"></a><a name="zh-cn_topic_0000002554424517_p783113012187"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000002554424517_row1272474920205"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000002554424517_p17301775812"><a name="zh-cn_topic_0000002554424517_p17301775812"></a><a name="zh-cn_topic_0000002554424517_p17301775812"></a><span id="zh-cn_topic_0000002554424517_ph2272194216543"><a name="zh-cn_topic_0000002554424517_ph2272194216543"></a><a name="zh-cn_topic_0000002554424517_ph2272194216543"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000002554424517_p37256491200"><a name="zh-cn_topic_0000002554424517_p37256491200"></a><a name="zh-cn_topic_0000002554424517_p37256491200"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section618mcpsimp"></a>

框架需要使用的workspace称之为系统workspace。[Matmul Kernel侧接口](Matmul-Kernel侧接口.md)等高阶API需要系统workspace，所以在使用该类API时，需要调用该接口，设置系统workspace的指针。采用工程化算子开发方式或者kernel直调方式（开启[HAVE\_WORKSPACE](基于样例工程完成Kernel直调.md#table481718169817)编译选项）时，不需要开发者手动设置，框架会自动设置。其他场景下，需要开发者调用SetSysWorkSpace进行设置。

在kernel侧调用该接口前，需要在host侧调用GetLibApiWorkSpaceSize获取系统workspace的大小，并在host侧设置workspacesize大小。样例如下：

```
// 用户自定义的tiling函数
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    AddApiTiling tiling;
    ...
    size_t usrSize = 256; // 设置用户需要使用的workspace大小。
    // 如需要使用系统workspace需要调用GetLibApiWorkSpaceSize获取系统workspace的大小。
    auto ascendcPlatform = platform_ascendc:: PlatformAscendC(context->GetPlatformInfo());
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t *currentWorkspace = context->GetWorkspaceSizes(1); // 通过框架获取workspace的指针，GetWorkspaceSizes入参为所需workspace的块数。当前限制使用一块。
    currentWorkspace[0] = usrSize + sysWorkspaceSize; // 设置总的workspace的数值大小，总的workspace空间由框架来申请并管理。
    ...
}
```

## 函数原型<a name="section620mcpsimp"></a>

```
__aicore__ inline void SetSysWorkspace(GM_ADDR workspace)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  接口参数说明

<a name="table1055216132132"></a>
<table><thead align="left"><tr id="row105531513121315"><th class="cellrowborder" valign="top" width="16.49%" id="mcps1.2.4.1.1"><p id="p5553171319138"><a name="p5553171319138"></a><a name="p5553171319138"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="11.93%" id="mcps1.2.4.1.2"><p id="p5553151313131"><a name="p5553151313131"></a><a name="p5553151313131"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="71.58%" id="mcps1.2.4.1.3"><p id="p655316136139"><a name="p655316136139"></a><a name="p655316136139"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row5553201314135"><td class="cellrowborder" valign="top" width="16.49%" headers="mcps1.2.4.1.1 "><p id="p8553813111314"><a name="p8553813111314"></a><a name="p8553813111314"></a>workspace</p>
</td>
<td class="cellrowborder" valign="top" width="11.93%" headers="mcps1.2.4.1.2 "><p id="p755318134134"><a name="p755318134134"></a><a name="p755318134134"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.58%" headers="mcps1.2.4.1.3 "><p id="p1185064715302"><a name="p1185064715302"></a><a name="p1185064715302"></a>核函数传入的workspace的指针，包括系统workspace和用户使用的workspace。</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section633mcpsimp"></a>

无

## 返回值说明<a name="section640mcpsimp"></a>

无

## 调用示例<a name="section642mcpsimp"></a>

```
template<typename aType, typename bType, typename cType, typename biasType>
__aicore__ inline void MatmulLeakyKernel<aType, bType, cType, biasType>::Init(
    GM_ADDR a, GM_ADDR b, GM_ADDR bias, GM_ADDR c, GM_ADDR workspace, const TCubeTiling& tiling, float alpha)
{
    // 融合算子的初始化操作
    // ...
    // workspace为核函数传入的GM指针，用于设置系统workspace
    AscendC::SetSysWorkspace(workspace);
    if (GetSysWorkSpacePtr() == nullptr) {
        return;
    }
    // 后续Matmul指令
    ...
}
```

