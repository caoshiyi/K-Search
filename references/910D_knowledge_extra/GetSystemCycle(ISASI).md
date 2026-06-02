# GetSystemCycle\(ISASI\)<a name="ZH-CN_TOPIC_0000002523344410"></a>

## 产品支持情况<a name="section1550532418810"></a>

<a name="table38301303189"></a>
<table><thead align="left"><tr id="row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p1883113061818"><a name="p1883113061818"></a><a name="p1883113061818"></a><span id="ph20833205312295"><a name="ph20833205312295"></a><a name="ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p783113012187"><a name="p783113012187"></a><a name="p783113012187"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row1272474920205"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p17301775812"><a name="p17301775812"></a><a name="p17301775812"></a><span id="ph2272194216543"><a name="ph2272194216543"></a><a name="ph2272194216543"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p37256491200"><a name="p37256491200"></a><a name="p37256491200"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section618mcpsimp"></a>

获取当前系统cycle数，若换算成时间需要按照50MHz的频率，时间单位为us，换算公式为：time = \(cycle数/50\) us 。

## 函数原型<a name="section620mcpsimp"></a>

```
__aicore__ inline int64_t GetSystemCycle()
```

## 参数说明<a name="section622mcpsimp"></a>

无

## 返回值说明<a name="section640mcpsimp"></a>

返回系统cycle数。

## 约束说明<a name="section633mcpsimp"></a>

该接口是PIPE\_S流水，若需要测试其他流水的指令时间，需要在调用该接口前通过[PipeBarrier](PipeBarrier(ISASI).md)插入对应流水的同步，具体请参考[调用示例](#li126441923175612)。

## 调用示例<a name="section177231425115410"></a>

-   如下示例通过GetSystemCycle获取系统cycle数，并换算成时间（单位：us）。

    ```
    #include "kernel_operator.h"
    
    __aicore__ inline void InitTilingParam(int32_t& totalSize, int32_t& loopSize)
    {
        int64_t systemCycleBefore = AscendC::GetSystemCycle(); // 调用GetBlockNum指令前的cycle数
        loopSize = totalSize / AscendC::GetBlockNum();
        int64_t systemCycleAfter = AscendC::GetSystemCycle(); // 调用GetBlockNum指令后的cycle数
        int64_t GetBlockNumCycle = systemCycleAfter - systemCycleBefore; // 执行GetBlockNum指令所用的cycle数
        int64_t CycleToTimeBase = 50; // cycle数转换成时间的基准单位，固定为50
        int64_t GetBlockNumTime = GetBlockNumCycle/CycleToTimeBase; // 执行GetBlockNum指令所用时间，单位为us
    };
    ```

-   <a name="li126441923175612"></a>如下示例为获取矢量计算Add指令时间的关键代码片段，在调用GetSystemCycle之前，插入了PIPE\_ALL同步，可以保证相关指令执行完后再获取cycle数。

    ```
    PipeBarrier<PIPE_ALL>();
    int64_t systemCycleBefore = AscendC::GetSystemCycle(); // 调用Add指令前的cycle数
    AscendC::Add(dstLocal, src0Local, src1Local, 512);
    PipeBarrier<PIPE_ALL>();
    int64_t systemCycleAfter = AscendC::GetSystemCycle(); // 调用Add指令后的cycle数
    int64_t GetBlockNumCycle = systemCycleAfter - systemCycleBefore; // 执行Add指令所用的cycle数
    int64_t CycleToTimeBase = 50; // cycle数转换成时间的基准单位，固定为50
    int64_t GetBlockNumTime = GetBlockNumCycle/CycleToTimeBase; // 执行Add指令所用时间，单位为us
    ```

