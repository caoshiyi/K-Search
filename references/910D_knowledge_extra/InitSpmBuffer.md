# InitSpmBuffer<a name="ZH-CN_TOPIC_0000002554424215"></a>

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

初始化SPM Buffer。

## 函数原型<a name="section620mcpsimp"></a>

-   暂存到workspace初始化，需要指定GM地址为SPM Buffer：

    ```
    template <typename T>
    __aicore__ inline void InitSpmBuffer(const GlobalTensor<T>& workspace, const int32_t bufferSize)
    ```

-   暂存到L1 Buffer初始化，不需要指定地址，会默认暂存到L1 Buffer，只需要传入需要的SPM Buffer大小：

    ```
    __aicore__ inline void InitSpmBuffer(const int32_t  bufferSize)
    ```

## 参数说明<a name="section622mcpsimp"></a>

<a name="table1794522316251"></a>
<table><thead align="left"><tr id="row19456238252"><th class="cellrowborder" valign="top" width="12.36%" id="mcps1.1.4.1.1"><p id="p119458239258"><a name="p119458239258"></a><a name="p119458239258"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="12.379999999999999%" id="mcps1.1.4.1.2"><p id="p9945152332514"><a name="p9945152332514"></a><a name="p9945152332514"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="75.26%" id="mcps1.1.4.1.3"><p id="p1594552312513"><a name="p1594552312513"></a><a name="p1594552312513"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row1694552372511"><td class="cellrowborder" valign="top" width="12.36%" headers="mcps1.1.4.1.1 "><p id="p1094516239250"><a name="p1094516239250"></a><a name="p1094516239250"></a>workspace</p>
</td>
<td class="cellrowborder" valign="top" width="12.379999999999999%" headers="mcps1.1.4.1.2 "><p id="p8945112312514"><a name="p8945112312514"></a><a name="p8945112312514"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.26%" headers="mcps1.1.4.1.3 "><p id="p169454239253"><a name="p169454239253"></a><a name="p169454239253"></a>workspace地址。</p>
</td>
</tr>
<tr id="row524916295111"><td class="cellrowborder" valign="top" width="12.36%" headers="mcps1.1.4.1.1 "><p id="p92491829141111"><a name="p92491829141111"></a><a name="p92491829141111"></a>bufferSize</p>
</td>
<td class="cellrowborder" valign="top" width="12.379999999999999%" headers="mcps1.1.4.1.2 "><p id="p6249142931114"><a name="p6249142931114"></a><a name="p6249142931114"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.26%" headers="mcps1.1.4.1.3 "><p id="p17249929131117"><a name="p17249929131117"></a><a name="p17249929131117"></a>SPM Buffer的大小，单位是字节。</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section633mcpsimp"></a>

无

## 返回值说明<a name="section640mcpsimp"></a>

无

## 调用示例<a name="section642mcpsimp"></a>

-   暂存到workspace初始化

    ```
    AscendC::TPipe pipe;
    int len = 1024; // 设置spm buffer为1024个类型为T的数据
    workspace_gm.SetGlobalBuffer((__gm__ T *)usrWorkspace, len);  // 此处的usrWorkspace为用户自定义的workspace
    auto gm = workspace_gm[AscendC::GetBlockIdx() * len];
    pipe.InitSpmBuffer(gm, len * sizeof(T));
    ```

-   暂存到L1 Buffer初始化

    ```
    AscendC::TPipe pipe;
    int len = 1024; // 设置spm buffer为1024个类型为T的数据
    pipe.InitSpmBuffer(len * sizeof(T));
    ```

