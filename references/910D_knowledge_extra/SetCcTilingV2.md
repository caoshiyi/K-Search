# SetCcTilingV2<a name="ZH-CN_TOPIC_0000002523344700"></a>

## 产品支持情况<a name="section1586581915393"></a>

<a name="table169596713360"></a>
<table><thead align="left"><tr id="row129590715369"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p17959971362"><a name="p17959971362"></a><a name="p17959971362"></a><span id="ph895914718367"><a name="ph895914718367"></a><a name="ph895914718367"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p89594763612"><a name="p89594763612"></a><a name="p89594763612"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row18959673369"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p1595910763613"><a name="p1595910763613"></a><a name="p1595910763613"></a><span id="ph1595918753613"><a name="ph1595918753613"></a><a name="ph1595918753613"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p1695957133611"><a name="p1695957133611"></a><a name="p1695957133611"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section1769414212182"></a>

用于设置HCCL客户端中某个通信算法配置的TilingData地址。

## 函数原型<a name="section14969112112188"></a>

```
__aicore__ inline int32_t SetCcTilingV2(uint64_t offset)
```

## 参数说明<a name="section12546122891815"></a>

**表 1**  接口参数说明

<a name="table11541249132419"></a>
<table><thead align="left"><tr id="row81541849152411"><th class="cellrowborder" valign="top" width="15.981598159815983%" id="mcps1.2.4.1.1"><p id="p715444932416"><a name="p715444932416"></a><a name="p715444932416"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="19.801980198019802%" id="mcps1.2.4.1.2"><p id="p1115410497248"><a name="p1115410497248"></a><a name="p1115410497248"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="64.2164216421642%" id="mcps1.2.4.1.3"><p id="p41549495249"><a name="p41549495249"></a><a name="p41549495249"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row11541349102415"><td class="cellrowborder" valign="top" width="15.981598159815983%" headers="mcps1.2.4.1.1 "><p id="p1615414972415"><a name="p1615414972415"></a><a name="p1615414972415"></a>offset</p>
</td>
<td class="cellrowborder" valign="top" width="19.801980198019802%" headers="mcps1.2.4.1.2 "><p id="p1615434911249"><a name="p1615434911249"></a><a name="p1615434911249"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="64.2164216421642%" headers="mcps1.2.4.1.3 "><p id="p11154194915243"><a name="p11154194915243"></a><a name="p11154194915243"></a>通信算法配置<a href="TilingData结构体.md#table678914014562">Mc2CcTiling</a>参数地址相对于<a href="TilingData结构体.md#table4835205712588">Mc2InitTiling</a>起始地址的偏移。<a href="TilingData结构体.md#table678914014562">Mc2CcTiling</a>在Host侧计算得出，具体请参考<a href="TilingData结构体.md#table678914014562">表2 Mc2CcTiling参数说明</a>，由框架传递到Kernel函数中使用。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section641993511814"></a>

-   HCCL\_SUCCESS，表示成功。
-   HCCL\_FAILED，表示失败。

## 约束说明<a name="section15931595196"></a>

-   若调用本接口，必须保证[InitV2](InitV2.md)在本接口前被调用。
-   Tiling参数相同的同一种通信算法在调用Prepare接口前，只需要调用一次本接口，请参考调用示例：[类型不同、Tiling参数不同的通信](#li71505119260)。
-   对于同一种通信算法，如果Tiling参数不同，重复调用本接口会覆盖之前的Tiling参数地址，因此需要在调用Prepare接口后再调用本接口设置新的Tiling参数。请参考调用示例：[类型相同、Tiling参数不同的通信](#li1163031215116)。
-   若调用本接口，必须使用标准C++语法定义TilingData结构体的开发方式，具体请参考[使用标准C++语法定义Tiling结构体](使用标准C++语法定义Tiling结构体.md)。

## 调用示例<a name="section7544820141919"></a>

-   用户自定义TilingData结构体：

    ```
    class UserCustomTilingData {
        AscendC::tiling::Mc2InitTiling initTiling;
        AscendC::tiling::Mc2CcTiling allGatherTiling;
        AscendC::tiling::Mc2CcTiling allReduceTiling1;
        AscendC::tiling::Mc2CcTiling allReduceTiling2;
        CustomTiling param;
    };
    ```

-   <a name="li71505119260"></a>类型不同、Tiling参数不同的通信

    ```
    extern "C" __global__ __aicore__ void userKernel(GM_ADDR aGM, GM_ADDR workspaceGM, GM_ADDR tilingGM) {
        REGISTER_TILING_DEFAULT(UserCustomTilingData);
        GET_TILING_DATA_WITH_STRUCT(UserCustomTilingData, tilingData, tilingGM);
    
        Hccl hccl;
        GM_ADDR contextGM = AscendC::GetHcclContext<0>();
        hccl.InitV2(contextGM, &tilingData);
    
        // 在下发任务之前，通过SetCcTilingV2设置对应的tiling
        if (hccl.SetCcTilingV2(offsetof(UserCustomTilingData, allGatherTiling)) != HCCL_SUCCESS ||
            hccl.SetCcTilingV2(offsetof(UserCustomTilingData, allReduceTiling1)) != HCCL_SUCCESS) {
            return;
        }
        const auto agHandleId = hccl.AllGather<true>(sendBuf, recvBuf, dataCount, HcclDataType::HCCL_DATA_TYPE_FP16);
        hccl.Wait(agHandleId);
    
        const auto arHandleId = hccl.AllReduce<true>(sendBuf, recvBuf, dataCount, HcclDataType::HCCL_DATA_TYPE_FP16, HcclReduceOp::HCCL_REDUCE_SUM);
        hccl.Wait(arHandleId);
    
        hccl.Finalize();
    }
    ```

-   <a name="li1163031215116"></a>类型相同、Tiling参数不同的通信

    ```
    extern "C" __global__ __aicore__ void userKernel(GM_ADDR aGM, GM_ADDR workspaceGM, GM_ADDR tilingGM) {
        REGISTER_TILING_DEFAULT(UserCustomTilingData);
        GET_TILING_DATA_WITH_STRUCT(UserCustomTilingData, tilingData, tilingGM);
    
        Hccl hccl;
        GM_ADDR contextGM = AscendC::GetHcclContext<0>();
        hccl.InitV2(contextGM, &tilingData);
    
        // 在下发通信任务之前，通过SetCcTilingV2设置对应的Tiling参数地址
        if (hccl.SetCcTilingV2(offsetof(UserCustomTilingData, allReduceTiling1)) != HCCL_SUCCESS) {
            return;
        }
        const auto arHandleId1 = hccl.AllReduce<true>(sendBuf, recvBuf, dataCount, HcclDataType::HCCL_DATA_TYPE_FP16, HcclReduceOp::HCCL_REDUCE_SUM);
        hccl.Wait(arHandleId1);
        
        // 第二次AllReduce的Tiling参数与第一次不同，在第一次Prepare之后再调用SetCcTilingV2
        if (hccl.SetCcTilingV2(offsetof(UserCustomTilingData, allReduceTiling2)) != HCCL_SUCCESS) {
            return;
        }
        const auto arHandleId2 = hccl.AllReduce<true>(sendBuf, recvBuf, dataCount, HcclDataType::HCCL_DATA_TYPE_FP16, HcclReduceOp::HCCL_REDUCE_SUM);
        hccl.Wait(arHandleId2);
    
        hccl.Finalize();
    }
    ```

