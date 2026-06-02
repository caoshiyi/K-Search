# AddPlatformInfo

**页面ID:** atlasascendc_api_07_00070  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_00070.html

---

#### 功能说明

设置硬件平台信息便于用户在算子Tiling函数调测中使用。支持以下两种设置方式：

- **自动获取当前硬件平台信息**：传入空指针，自动获取当前硬件信息并添加到ContextBuilder类中。
- **指定硬件平台信息**：传入具体的昇腾AI处理器型号，添加对应硬件信息至ContextBuilder类中。

若设置失败，会打印报错信息。关于日志配置和查看，请参考日志。

#### 函数原型

```
ContextBuilder &AddPlatformInfo(const char* customSocVersion)
```

#### 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| customSocVersion | 输入 | 昇腾AI处理器型号。配置方式如下：                     - 针对如下产品：在安装昇腾AI处理器的服务器执行**npu-smi info**命令进行查询，获取**Name**信息。实际配置值为AscendName，例如**Name**取值为*xxxyy*，实际配置值为Ascend*xxxyy*。                           Atlas A2 训练系列产品              /               Atlas A2 推理系列产品                                         Atlas 200I/500 A2 推理产品                                         Atlas 推理系列产品                                         Atlas 训练系列产品                         - 针对如下产品，在安装昇腾AI处理器的服务器执行**npu-smi info -t board -i ***id*** -c ***chip_id*命令进行查询，获取**Chip Name**和**NPU Name**信息，实际配置值为Chip Name_NPU Name。例如**Chip Name**取值为Ascend*xxx*，**NPU Name**取值为1234，实际配置值为Ascend*xxx**_*1234。其中：                           - id：设备id，通过**npu-smi info -l**命令查出的NPU ID即为设备id。               - chip_id：芯片id，通过**npu-smi info -m**命令查出的Chip ID即为芯片id。                                       Atlas A3 训练系列产品              /               Atlas A3 推理系列产品 |

#### 返回值说明

当前ContextBuilder对象。

#### 约束说明

AddPlatformInfo调用后需要通过BuildTilingContext来构建Tiling的上下文，并传递给Tiling函数来使用。

#### 调用示例

```
void AddPlatformInfoDemo(......)
{
    auto holder = context_ascendc::ContextBuilder()
	// ... ... // 增加算子输入输出接口的调用
	.AddPlatformInfo("Ascendxxxyy")
	.BuildTilingContext();
    auto tilingContext = holder->GetContext<gert::TilingContext>();
    // ... ...
}
```
