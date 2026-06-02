# GetCoreNum<a name="ZH-CN_TOPIC_0000002523343532"></a>

## 功能说明<a name="zh-cn_topic_0000001647201621_zh-cn_topic_0000001442758437_section36583473819"></a>

获取当前硬件平台的核数。若AI Core的架构为Cube、Vector分离模式，返回Vector Core的核数；耦合模式返回AI Core的核数。

## 函数原型<a name="zh-cn_topic_0000001647201621_zh-cn_topic_0000001442758437_section13230182415108"></a>

```
uint32_t GetCoreNum(void) const
```

## 参数说明<a name="zh-cn_topic_0000001647201621_zh-cn_topic_0000001442758437_section189014013619"></a>

无

## 返回值说明<a name="zh-cn_topic_0000001647201621_zh-cn_topic_0000001442758437_section25791320141317"></a>

Ascend 950PR/Ascend 950DT，分离模式，返回Vector Core的核数

## 约束说明<a name="zh-cn_topic_0000001647201621_zh-cn_topic_0000001442758437_section19165124931511"></a>

无

## 调用示例<a name="zh-cn_topic_0000001647201621_zh-cn_topic_0000001442758437_section320753512363"></a>

```
ge::graphStatus TilingXXX(gert::TilingContext* context) {
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto coreNum = ascendcPlatform.GetCoreNum();
    // ... 根据核数自行设计Tiling策略
    context->SetBlockDim(coreNum);
    return ret;
}
```

