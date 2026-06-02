# SetBlockDim

**页面ID:** atlasopapi_07_00236  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00236.html

---

#### 函数功能

设置blockDim，即参与计算的Vector或者Cube核数。

#### 函数原型

**ge::graphStatus SetBlockDim(const uint32_t block_dim)**

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| block_dim | 输入 | blockDim是逻辑核的概念，取值范围为[1,65535]。为了充分利用硬件资源，一般设置为物理核的核数或其倍数。                     - 对于耦合模式和分离模式，blockDim在运行时的意义和设置规则有一些区别，具体说明如下：                            - 耦合模式：由于其Vector、Cube单元是集成在一起的，blockDim用于设置启动多个AICore核实例执行，不区分Vector、Cube。AI Core的核数可以通过GetCoreNumAiv或者GetCoreNumAic获取。               - 分离模式                                  - 针对仅包含Vector计算的算子，blockDim用于设置启动多少个Vector（AIV）实例执行，比如某款AI处理器上有40个Vector核，建议设置为40。                   - 针对仅包含Cube计算的算子，blockDim用于设置启动多少个Cube（AIC）实例执行，比如某款AI处理器上有20个Cube核，建议设置为20。                   - 针对Vector/Cube融合计算的算子，启动时，按照AIV和AIC组合启动，blockDim用于设置启动多少个组合执行，比如某款AI处理器上有40个Vector核和20个Cube核，一个组合是2个Vector核和1个Cube核，建议设置为20，此时会启动20个组合，即40个Vector核和20个Cube核。**注意：该场景下，设置的blockDim逻辑核的核数不能超过物理核（2个Vector核和1个Cube核组合为1个物理核）的核数。**                   - AIC/AIV的核数分别通过GetCoreNumAic和GetCoreNumAiv接口获取。                                     - 在设置Device资源限制的场景下，设置的blockDim核数不能超过通过GetCoreNumAiv等接口获取的物理核数。例如，如果使用aclrtSetStreamResLimit设置Stream级别的Device资源限制为8个核，那么blockDim不能超过8，否则会抢占其他Stream的资源，导致资源限制失效。           - 如果开发者使用了Device资源限制特性，那么算子设置的blockDim不应超过PlatformAscendC提供核数的API（GetCoreNum/GetCoreNumAic/GetCoreNumAiv等）返回的核数。例如，使用aclrtSetStreamResLimit设置Stream级别的Vector核数为8，那么GetCoreNumAiv接口返回值为8，针对Vector算子设置的blockDim不应超过8，否则会抢占其他Stream的资源，导致资源限制失效。 |

#### 返回值说明

设置成功时返回“ge::GRAPH_SUCCESS”。

关于graphStatus的定义，请参见ge::graphStatus。

#### 约束说明

无。

#### 调用示例

```
ge::graphStatus Tiling4XXX(TilingContext* context) {
  auto ret = context->SetBlockDim(32);
  // ...
}
```
