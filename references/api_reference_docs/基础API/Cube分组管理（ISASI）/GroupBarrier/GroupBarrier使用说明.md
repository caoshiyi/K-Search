# GroupBarrier使用说明

**页面ID:** atlasascendc_api_07_0302  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0302.html

---

当同一个CubeResGroupHandle中的两个AIV任务之间存在依赖关系时，可以使用GroupBarrier控制同步。假设一组AIV A做完任务x以后，另外一组AIV B才可以开始后续业务，称AIV A组为Arrive组，AIV B组为Wait组。

基于GroupBarrier的组同步使用步骤如下：

1. 创建GroupBarrier。
2. 被等待的AIV调用Arrive，需要等待的AIV调用Wait。

下文仅提供示例代码片段，更多完整样例请参考[GroupBarrier样例](https://gitee.com/ascend/samples/tree/master/operator/ascendc/2_features/16_group_barrier)。

1. 创建GroupBarrier。

```
constexpr int32_t ARRIVE_NUM = 2; // Arrive组的AIV个数
constexpr int32_t WAIT_NUM = 6; // Wait组的AIV个数
AscendC::GroupBarrier<AscendC::PipeMode::MTE3_MODE> barA(workspace, ARRIVE_NUM, WAIT_NUM);  // 创建GroupBarrier，用户自行管理并对这部分workspace清零
```

2. 被等待的AIV调用Arrive，需要等待的AIV调用Wait。

```
auto id = AscendC::GetBlockIdx();
if (id > 0 && id < ARRIVE_NUM) {
  //各种Vector计算逻辑，用户自行实现
  barA.Arrive(id);
} else(id >= ARRIVE_NUM && id < ARRIVE_NUM + WAIT_NUM){
  barA.Wait(id - ARRIVE_NUM);
  // 各种Vector计算逻辑，用户自行实现
}
```
