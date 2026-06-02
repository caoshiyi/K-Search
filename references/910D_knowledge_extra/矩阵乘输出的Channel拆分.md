# 矩阵乘输出的Channel拆分<a name="ZH-CN_TOPIC_0000002523289068"></a>

## 功能介绍<a name="zh-cn_topic_0000002264134832_section310824820358"></a>

矩阵乘输出的Channel拆分，又称ChannelSplit。指当Matmul计算结果C矩阵的格式为[NZ](基础知识.md#zh-cn_topic_0000001622194138_section1453415011)时，C矩阵采用分形存储，关于NZ格式的详细内容请参考[数据格式](基础知识.md#zh-cn_topic_0000001622194138_section1453415011)。当C矩阵的物理排布格式为NZ、数据类型为float时，默认情况下，每个分形内部包含16\*16个元素，即分形的大小为16\*16。ChannelSplit的功能为将此场景下C矩阵的每个16\*16的分形切分为16\*8的分形，使得C矩阵按照16\*8的分形进行存储。

由于1个float类型数据的大小为4字节，16\*8的分形在内轴满足32字节对齐，内轴上的数据量与一条NPU矢量计算指令处理的数据单元一致，这便于后续的其它计算。ChannelSplit功能默认不启用，用户需通过设置[MatmulConfig](MatmulConfig.md#table1761013213153)中的isEnableChannelSplit参数为true来开启此功能。

**图 1**  ChannelSplit功能示意图<a name="zh-cn_topic_0000002264134832_fig38211632121711"></a>  
<!-- img2text -->
```text
                         非ChannelSplit                                             ChannelSplit

                  <────16────>                                                <──8──>
        <────────────────────────64────────────────────────>      <────────────────────────64────────────────────────>
      ↑                                                            ↑
      │16                                                          │16
      │                                                            │
      │   ┌──────────┬──────────┬──────────┬──────────┐             │   ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
      │   │ ↗───↘    │  ↗───↘   │  ↗───↘   │  ↗───↘   │             │   │↗─↘  │↗─↘  │↗─↘  │↗─↘  │↗─↘  │↗─↘  │↗─↘  │↗─↘  │
      │   │ ↖───→    │  ↖───→   │  ↖───→   │  ↖───→   │             │   │↖─→  │↖─→  │↖─→  │↖─→  │↖─→  │↖─→  │↖─→  │↖─→  │
      │   │ ↗───↘    │  ↗───↘   │  ↗───↘   │  ↗───↘   │             │   │↗─↘  │↗─↘  │↗─↘  │↗─↘  │↗─↘  │↗─↘  │↗─↘  │↗─↘  │
      │   │ ↖───→    │  ↖───→   │  ↖───→   │  ↖───→   │             │   │↖─→  │↖─→  │↖─→  │↖─→  │↖─→  │↖─→  │↖─→  │↖─→  │
      │   ├──────────┼──────────┼──────────┼──────────┤             │   ├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤
  64  │   │ ↗───↘    │  ↗───↘   │  ↗───↘   │  ↗───↘   │         64  │   │↗─↘  │↗─↘  │↗─↘  │↗─↘  │↗─↘  │↗─↘  │↗─↘  │↗─↘  │
      │   │ ↖───→    │  ↖───→   │  ↖───→   │  ↖───→   │             │   │↖─→  │↖─→  │↖─→  │↖─→  │↖─→  │↖─→  │↖─→  │↖─→  │
      │   │ ↗───↘    │  ↗───↘   │  ↗───↘   │  ↗───↘   │             │   │↗─↘  │↗─↘  │↗─↘  │↗─↘  │↗─↘  │↗─↘  │↗─↘  │↗─↘  │
      │   │ ↖───→    │  ↖───→   │  ↖───→   │  ↖───→   │             │   │↖─→  │↖─→  │↖─→  │↖─→  │↖─→  │↖─→  │↖─→  │↖─→  │
      ↓   └──────────┴──────────┴──────────┴──────────┘             ↓   └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘

                        [ 矩阵C ]   [ NZ(16,16) ]                                  [ 矩阵C ]   [ NZ(16,8) ]
```

说明:
- 左图标题为“非ChannelSplit”，右图标题为“ChannelSplit”。
- 两图整体高度均标注为 64，顶部整体宽度均标注为 64。
- 左图单个分形宽度标注为 16，对应 4 个 NZ(16,16) 分形并排。
- 右图单个分形宽度标注为 8，对应 8 个 NZ(16,8) 分形并排。
- 图中每个小框内的折线箭头表示数据在分形内部的排布/访问方向；复杂折线已做近似示意。

## 使用场景<a name="zh-cn_topic_0000002264134832_section118051016163613"></a>

对于NZ格式、float类型的C矩阵，需要按16\*8的分形存储时，使用该功能。

## 约束说明<a name="zh-cn_topic_0000002264134832_section14160134220363"></a>

开启ChannelSplit功能需满足：

-   C矩阵的数据排布格式为CubeFormat::NZ。
-   C矩阵的数据类型为float。
-   C矩阵的内存逻辑位置为Global Memory。
-   矩阵乘结果CO1数据类型为float。

## 调用示例<a name="zh-cn_topic_0000002264134832_section15486294368"></a>

完整的算子样例请参考[matmul\_channelsplit算子样例](https://gitee.com/ascend/ascendc-api-adv/tree/master/examples/matrix/matmul_channelsplit)。

```
// 指定获取和修改的MatmulConfig模板
constexpr static MatmulConfigMode configMode = MatmulConfigMode::CONFIG_NORM;
// 修改模板参数isEnableChannelSplit=true，开启该MatmulConfig模板的ChannelSplit功能
constexpr static MatmulFuncParams funcParamsChannelSplit{
    false, false, false, false, 0, IterateOrder::ORDER_M, ScheduleType::INNER_PRODUCT, true, false, false, false, true/*isEnableChannelSplit*/
};
constexpr static MatmulConfig MM_CFG = GetMMConfig<configMode>(funcParamsChannelSplit);
Matmul<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG> mm;

// 常规Matmul计算，最后输出分形为16*8
REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm);
mm.SetTensorA(gm_a);
mm.SetTensorB(gm_b);
mm.SetBias(gm_bias);
mm.IterateAll(gm_c);
```

