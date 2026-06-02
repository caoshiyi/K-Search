# AddPlatformInfo<a name="ZH-CN_TOPIC_0000002554423437"></a>

## 功能说明<a name="zh-cn_topic_0000001867409741_zh-cn_topic_0000001389787297_section36583473819"></a>

设置硬件平台信息便于用户在算子Tiling函数调测中使用。支持以下两种设置方式：

-   **自动获取当前硬件平台信息**：传入空指针，自动获取当前硬件信息并添加到ContextBuilder类中。
-   **指定硬件平台信息**：传入具体的AI处理器型号，添加对应硬件信息至ContextBuilder类中。

若设置失败，会打印报错信息。关于日志配置和查看，请参考《环境变量参考》中“辅助功能 \> 日志”章节。

## 函数原型<a name="zh-cn_topic_0000001867409741_zh-cn_topic_0000001389787297_section13230182415108"></a>

```
ContextBuilder &AddPlatformInfo(const char* customSocVersion)
```

## 参数说明<a name="zh-cn_topic_0000001867409741_zh-cn_topic_0000001389787297_section75395119104"></a>

<a name="zh-cn_topic_0000001867409741_zh-cn_topic_0000001389787297_table111938719446"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001867409741_zh-cn_topic_0000001389787297_row6223476444"><th class="cellrowborder" valign="top" width="17.22%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001867409741_zh-cn_topic_0000001389787297_p10223674448"><a name="zh-cn_topic_0000001867409741_zh-cn_topic_0000001389787297_p10223674448"></a><a name="zh-cn_topic_0000001867409741_zh-cn_topic_0000001389787297_p10223674448"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="15.340000000000002%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001867409741_zh-cn_topic_0000001389787297_p645511218169"><a name="zh-cn_topic_0000001867409741_zh-cn_topic_0000001389787297_p645511218169"></a><a name="zh-cn_topic_0000001867409741_zh-cn_topic_0000001389787297_p645511218169"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="67.44%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001867409741_zh-cn_topic_0000001389787297_p1922337124411"><a name="zh-cn_topic_0000001867409741_zh-cn_topic_0000001389787297_p1922337124411"></a><a name="zh-cn_topic_0000001867409741_zh-cn_topic_0000001389787297_p1922337124411"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001867409741_zh-cn_topic_0000001389787297_row152234713443"><td class="cellrowborder" valign="top" width="17.22%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001867409741_zh-cn_topic_0000001389787297_p8563195616313"><a name="zh-cn_topic_0000001867409741_zh-cn_topic_0000001389787297_p8563195616313"></a><a name="zh-cn_topic_0000001867409741_zh-cn_topic_0000001389787297_p8563195616313"></a>customSocVersion</p>
</td>
<td class="cellrowborder" valign="top" width="15.340000000000002%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001867409741_zh-cn_topic_0000001389787297_p15663137127"><a name="zh-cn_topic_0000001867409741_zh-cn_topic_0000001389787297_p15663137127"></a><a name="zh-cn_topic_0000001867409741_zh-cn_topic_0000001389787297_p15663137127"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.44%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001867409741_zh-cn_topic_0000001389787297_p2684123934216"><a name="zh-cn_topic_0000001867409741_zh-cn_topic_0000001389787297_p2684123934216"></a><a name="zh-cn_topic_0000001867409741_zh-cn_topic_0000001389787297_p2684123934216"></a><span id="ph1117616368528"><a name="ph1117616368528"></a><a name="ph1117616368528"></a>AI处理器</span>型号。</p>
<a name="ul1124912113117"></a><a name="ul1124912113117"></a><ul id="ul1124912113117"><li>针对如下<span id="zh-cn_topic_0000002554324119_ph13446185695413"><a name="zh-cn_topic_0000002554324119_ph13446185695413"></a><a name="zh-cn_topic_0000002554324119_ph13446185695413"></a>产品</span>，在安装<span id="zh-cn_topic_0000002554324119_ph17911124171120"><a name="zh-cn_topic_0000002554324119_ph17911124171120"></a><a name="zh-cn_topic_0000002554324119_ph17911124171120"></a>AI处理器</span>的服务器执行<strong id="zh-cn_topic_0000002554324119_zh-cn_topic_0000001264656721_zh-cn_topic_0000001117597244_b206066255591"><a name="zh-cn_topic_0000002554324119_zh-cn_topic_0000001264656721_zh-cn_topic_0000001117597244_b206066255591"></a><a name="zh-cn_topic_0000002554324119_zh-cn_topic_0000001264656721_zh-cn_topic_0000001117597244_b206066255591"></a>npu-smi info -t board -i </strong><em id="zh-cn_topic_0000002554324119_zh-cn_topic_0000001264656721_zh-cn_topic_0000001117597244_i16609202515915"><a name="zh-cn_topic_0000002554324119_zh-cn_topic_0000001264656721_zh-cn_topic_0000001117597244_i16609202515915"></a><a name="zh-cn_topic_0000002554324119_zh-cn_topic_0000001264656721_zh-cn_topic_0000001117597244_i16609202515915"></a>id</em><strong id="zh-cn_topic_0000002554324119_zh-cn_topic_0000001264656721_zh-cn_topic_0000001117597244_b14358631175910"><a name="zh-cn_topic_0000002554324119_zh-cn_topic_0000001264656721_zh-cn_topic_0000001117597244_b14358631175910"></a><a name="zh-cn_topic_0000002554324119_zh-cn_topic_0000001264656721_zh-cn_topic_0000001117597244_b14358631175910"></a> -c </strong><em id="zh-cn_topic_0000002554324119_zh-cn_topic_0000001264656721_zh-cn_topic_0000001117597244_i16269732165915"><a name="zh-cn_topic_0000002554324119_zh-cn_topic_0000001264656721_zh-cn_topic_0000001117597244_i16269732165915"></a><a name="zh-cn_topic_0000002554324119_zh-cn_topic_0000001264656721_zh-cn_topic_0000001117597244_i16269732165915"></a>chip_id</em>命令进行查询，获取<strong id="zh-cn_topic_0000002554324119_b11257114917192"><a name="zh-cn_topic_0000002554324119_b11257114917192"></a><a name="zh-cn_topic_0000002554324119_b11257114917192"></a>Chip Name</strong>和<strong id="zh-cn_topic_0000002554324119_b72671651121916"><a name="zh-cn_topic_0000002554324119_b72671651121916"></a><a name="zh-cn_topic_0000002554324119_b72671651121916"></a>NPU Name</strong>信息，实际配置值为Chip Name_NPU Name。例如<strong id="zh-cn_topic_0000002554324119_b13136111611203"><a name="zh-cn_topic_0000002554324119_b13136111611203"></a><a name="zh-cn_topic_0000002554324119_b13136111611203"></a>Chip Name</strong>取值为Ascend<em id="zh-cn_topic_0000002554324119_i68701996189"><a name="zh-cn_topic_0000002554324119_i68701996189"></a><a name="zh-cn_topic_0000002554324119_i68701996189"></a>xxx</em>，<strong id="zh-cn_topic_0000002554324119_b51347352112"><a name="zh-cn_topic_0000002554324119_b51347352112"></a><a name="zh-cn_topic_0000002554324119_b51347352112"></a>NPU Name</strong>取值为1234，实际配置值为Ascend<em id="zh-cn_topic_0000002554324119_i82901912141813"><a name="zh-cn_topic_0000002554324119_i82901912141813"></a><a name="zh-cn_topic_0000002554324119_i82901912141813"></a>xxx</em><em id="zh-cn_topic_0000002554324119_i154501458102213"><a name="zh-cn_topic_0000002554324119_i154501458102213"></a><a name="zh-cn_topic_0000002554324119_i154501458102213"></a>_</em>1234。其中：<a name="zh-cn_topic_0000002554324119_ul2747601334"></a><a name="zh-cn_topic_0000002554324119_ul2747601334"></a><ul id="zh-cn_topic_0000002554324119_ul2747601334"><li>id：设备id，通过<strong id="zh-cn_topic_0000002554324119_b83171930133314"><a name="zh-cn_topic_0000002554324119_b83171930133314"></a><a name="zh-cn_topic_0000002554324119_b83171930133314"></a>npu-smi info -l</strong>命令查出的NPU ID即为设备id。</li><li>chip_id：芯片id，通过<strong id="zh-cn_topic_0000002554324119_b18888204343317"><a name="zh-cn_topic_0000002554324119_b18888204343317"></a><a name="zh-cn_topic_0000002554324119_b18888204343317"></a>npu-smi info -m</strong>命令查出的Chip ID即为芯片id。</li></ul>
<p id="zh-cn_topic_0000002554324119_p1790216395447"><a name="zh-cn_topic_0000002554324119_p1790216395447"></a><a name="zh-cn_topic_0000002554324119_p1790216395447"></a><span id="zh-cn_topic_0000002554324119_ph2272194216543"><a name="zh-cn_topic_0000002554324119_ph2272194216543"></a><a name="zh-cn_topic_0000002554324119_ph2272194216543"></a>Ascend 950PR/Ascend 950DT</span></p>
</li></ul>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0000001867409741_zh-cn_topic_0000001389787297_section25791320141317"></a>

当前ContextBuilder对象。

## 约束说明<a name="zh-cn_topic_0000001867409741_zh-cn_topic_0000001389787297_section19165124931511"></a>

AddPlatformInfo调用后需要通过[BuildTilingContext](BuildTilingContext.md)来构建Tiling的上下文，并传递给Tiling函数来使用。

## 调用示例<a name="zh-cn_topic_0000001867409741_zh-cn_topic_0000001389787297_section320753512363"></a>

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

