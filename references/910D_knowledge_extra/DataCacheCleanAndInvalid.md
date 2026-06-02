# DataCacheCleanAndInvalid<a name="ZH-CN_TOPIC_0000002554424081"></a>

## 产品支持情况<a name="section1550532418810"></a>

<a name="table18341227114216"></a>
<table><thead align="left"><tr id="row1834122714219"><th class="cellrowborder" valign="top" width="53.64%" id="mcps1.1.4.1.1"><p id="p83416275429"><a name="p83416275429"></a><a name="p83416275429"></a><span id="ph93442717421"><a name="ph93442717421"></a><a name="ph93442717421"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="24.63%" id="mcps1.1.4.1.2"><p id="p103412764211"><a name="p103412764211"></a><a name="p103412764211"></a>是否支持（<span>支持配置dcciDst</span>的原型）</p>
</th>
<th class="cellrowborder" align="center" valign="top" width="21.73%" id="mcps1.1.4.1.3"><p id="p107301733174216"><a name="p107301733174216"></a><a name="p107301733174216"></a><span>是否支持（不支持配置dcciDst的原型）</span></p>
</th>
</tr>
</thead>
<tbody><tr id="row1272474920205"><td class="cellrowborder" valign="top" width="53.64%" headers="mcps1.1.4.1.1 "><p id="p17301775812"><a name="p17301775812"></a><a name="p17301775812"></a><span id="ph2272194216543"><a name="ph2272194216543"></a><a name="ph2272194216543"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="24.63%" headers="mcps1.1.4.1.2 "><p id="p37256491200"><a name="p37256491200"></a><a name="p37256491200"></a>√</p>
</td>
<td class="cellrowborder" align="center" valign="top" width="21.73%" headers="mcps1.1.4.1.3 "><p id="p6730933174216"><a name="p6730933174216"></a><a name="p6730933174216"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section618mcpsimp"></a>

在AI Core内部，Scalar单元和DMA单元都可能对Global Memory进行访问。

**图 1**  DataCache内存层次示意图<a name="fig1161014168448"></a>  
<!-- img2text -->
```text
AI Core
┌──────────────────────────────────────────────────────────────────────────────┐
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                                  AIC1                                  │  │
│  │  ┌──────────────────────────────────────────────────────────────────┐  │  │
│  │  │                               AIC 0                              │  │  │
│  │  │  ┌────────────────────────────────────────────────────────────┐  │  │  │
│  │  │  │                           AIV 1                            │  │  │  │
│  │  │  │                                                            │  │  │  │
│  │  │  │                                AIV 0                       │  │  │  │
│  │  │  │                                                            │  │  │  │
│  │  │  │  ┌────────────┐      ┌──────────────────┐      ┌────────┐ │  │  │  │
│  │  │  │  │ ScalarUnit │      │        UB        │ ←──→ │  DMA   │ │  │  │  │
│  │  │  │  └────────────┘      └──────────────────┘      └────────┘ │  │  │  │
│  │  │  │        ↑                                               ↑   │  │  │  │
│  │  │  │        ↓                                               │   │  │  │  │
│  │  │  │                                                            │  │  │  │
│  │  │  │                    DataCache                               │  │  │  │
│  │  │  │  ┌──────────────┬──────────────┬──────────────┬──────────┐ │  │  │  │
│  │  │  │  │  cacheline   │  cacheline   │  cacheline   │          │ │  │  │  │
│  │  │  │  │   ( 64B )    │   ( 64B )    │   ( 64B )    │          │ │  │  │  │
│  │  │  │  └──────────────┴──────────────┴──────────────┴──────────┘ │  │  │  │
│  │  │  └────────────────────────────────────────────────────────────┘  │  │  │
│  │  └──────────────────────────────────────────────────────────────────┘  │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘
        ↑                                                   ↑
        ↓                                                   ↓
                         L2 Cache
┌──────────────┬──────────────┬──────────────┬──────────────┬──────────────┐
│  cacheline   │  cacheline   │  cacheline   │  cacheline   │              │
└──────────────┴──────────────┴──────────────┴──────────────┴──────────────┘
        ↑                                                   ↑
        ↓                                                   ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│                                GLobal Memory                                │
└──────────────────────────────────────────────────────────────────────────────┘
```

如上图所示：

-   DMA搬运单元读写Global Memory，数据通过DataCopy等接口在UB等Local Memory和Global Memory间交互，没有Cache一致性问题；
-   Scalar单元访问Global Memory，首先会访问每个核内的Data Cache，因此存在Data Cache与Global Memory的Cache一致性问题。

该接口用来刷新Cache，保证Cache的一致性，使用场景如下：

-   读取Global Memory的数据，但该数据可能在外部被其余核修改，此时需要使用DataCacheCleanAndInvalid接口，直接访问Global Memory，获取最新数据；

-   用户通过Scalar单元写Global Memory的数据，希望立刻写出，也需要使用DataCacheCleanAndInvalid接口。
-   针对Ascend 950PR/Ascend 950DT，原子操作过程中，如果希望改变后续数据的饱和模式，需要先使用DataCacheCleanAndInvalid接口将Cache Line中现存的数据立刻写出，再调用[SetCtrlSpr](SetCtrlSpr(ISASI).md)设置后续数据的饱和模式。

## 函数原型<a name="section620mcpsimp"></a>

-   支持通过配置dcciDst确保Data Cache与GM存储的一致性

    ```
    template <typename T, CacheLine entireType, DcciDst dcciDst>
    __aicore__ inline void DataCacheCleanAndInvalid(const GlobalTensor<T>& dst)
    ```

-   支持通过配置dcciDst确保Data Cache与Local Memory存储的一致性

    ```
    template <typename T, CacheLine entireType, DcciDst dcciDst>
    __aicore__ inline void DataCacheCleanAndInvalid(const LocalTensor<T>& dst)
    ```

-   不支持配置dcciDst，仅支持保证Data Cache与GM的一致性

    ```
    template <typename T, CacheLine entireType>
    __aicore__ inline void DataCacheCleanAndInvalid(const GlobalTensor<T>& dst)
    ```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table22921157942"></a>
<table><thead align="left"><tr id="row142921657844"><th class="cellrowborder" valign="top" width="18.14%" id="mcps1.2.3.1.1"><p id="p229219578419"><a name="p229219578419"></a><a name="p229219578419"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="81.86%" id="mcps1.2.3.1.2"><p id="p329218578415"><a name="p329218578415"></a><a name="p329218578415"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row963114014513"><td class="cellrowborder" valign="top" width="18.14%" headers="mcps1.2.3.1.1 "><p id="p963154019510"><a name="p963154019510"></a><a name="p963154019510"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="81.86%" headers="mcps1.2.3.1.2 "><p id="p06318408513"><a name="p06318408513"></a><a name="p06318408513"></a>dst的数据类型。</p>
</td>
</tr>
<tr id="row829205713419"><td class="cellrowborder" valign="top" width="18.14%" headers="mcps1.2.3.1.1 "><p id="p13292175719415"><a name="p13292175719415"></a><a name="p13292175719415"></a>entireType</p>
</td>
<td class="cellrowborder" valign="top" width="81.86%" headers="mcps1.2.3.1.2 "><p id="p12921357941"><a name="p12921357941"></a><a name="p12921357941"></a>指令操作的模式：</p>
<p id="p122921057547"><a name="p122921057547"></a><a name="p122921057547"></a>SINGLE_CACHE_LINE：只刷新传入地址所在的Cache Line，<strong id="b1429295716411"><a name="b1429295716411"></a><a name="b1429295716411"></a>注意如果该地址非64B对齐，只会操作传入地址到64B对齐的部分。</strong></p>
<p id="p62924571048"><a name="p62924571048"></a><a name="p62924571048"></a>ENTIRE_DATA_CACHE：此时传入的地址无效，核内会刷新整个Data Cache，但是耗时较大，<strong id="b42932571842"><a name="b42932571842"></a><a name="b42932571842"></a>性能敏感的场景慎用</strong>。</p>
</td>
</tr>
<tr id="row429315710413"><td class="cellrowborder" valign="top" width="18.14%" headers="mcps1.2.3.1.1 "><p id="p172937571143"><a name="p172937571143"></a><a name="p172937571143"></a>dcciDst</p>
</td>
<td class="cellrowborder" valign="top" width="81.86%" headers="mcps1.2.3.1.2 "><p id="p172935571246"><a name="p172935571246"></a><a name="p172935571246"></a>表示使用该接口来保证Data Cache与哪一种存储保持一致性，类型为DcciDst枚举类。</p>
<a name="ul0740116780"></a><a name="ul0740116780"></a><ul id="ul0740116780"><li>CACHELINE_ALL：与CACHELINE_OUT效果一致。</li><li>CACHELINE_UB：表示通过该接口来保证Data Cache与UB的一致性。</li><li>CACHELINE_OUT：表示通过该接口来保证Data Cache与Global Memory的一致性。</li><li>CACHELINE_ATOMIC：<a name="ul174370221398"></a><a name="ul174370221398"></a><ul id="ul174370221398"><li><span id="ph142607321878"><a name="ph142607321878"></a><a name="ph142607321878"></a>Ascend 950PR/Ascend 950DT</span>，原子操作过程中保证Data Cache和Global Memory的一致性。</li></ul>
</li></ul>
</td>
</tr>
</tbody>
</table>

**表 2**  参数说明

<a name="zh-cn_topic_0235751031_table33761356"></a>
<table><thead align="left"><tr id="zh-cn_topic_0235751031_row27598891"><th class="cellrowborder" valign="top" width="18.54%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="10.05%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="71.41%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row42461942101815"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p479605232211"><a name="p479605232211"></a><a name="p479605232211"></a>dst</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p044110221282"><a name="p044110221282"></a><a name="p044110221282"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p1179555214221"><a name="p1179555214221"></a><a name="p1179555214221"></a>需要刷新Cache的Tensor。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

无

## 调用示例<a name="section837496171220"></a>

```
// 示例1：SINGLE_CACHE_LINE 模式，假设mmAddr_为0x40（64B对齐）
AscendC::GlobalTensor<uint64_t> global;
global.SetGlobalBuffer((__gm__ uint64_t*)mmAddr_ + AscendC::GetBlockIdx() * 1024);
for( int i = 0; i < 8; i++) {
   global.SetValue(i, AscendC::GetBlockIdx());
}
// 由于首地址64B对齐，调用DataCacheCleanAndInvalid指令后，会立刻刷新前8个数
AscendC::DataCacheCleanAndInvalid<uint64_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(global);
// 示例2：SINGLE_CACHE_LINE 模式，假设mmAddr_为0x20（非64B对齐）
AscendC::GlobalTensor<uint64_t> global;
global.SetGlobalBuffer((__gm__ uint64_t*)mmAddr_ + AscendC::GetBlockIdx() * 1024);
for( int i = 0; i < 8; i++) {
   global.SetValue(i, AscendC::GetBlockIdx());
}
// 由于首地址非64B对齐，调用1条指令，只会刷新起始地址至64B字节对齐的部分，即前4个数
AscendC::DataCacheCleanAndInvalid<uint64_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(global);
// 需要再次调用DataCacheCleanAndInvalid指令，刷新后4个数
AscendC::DataCacheCleanAndInvalid<uint64_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(global[4]);
// 示例3：SINGLE_CACHE_LINE 模式，假设mmAddr_为0x40（64B对齐），多核处理场景（本样例仅做示例说明，便于开发者理解使用限制，非正常使用样例）
AscendC::GlobalTensor<uint64_t> global;
global.SetGlobalBuffer((__gm__ uint64_t*)mmAddr_);
global.SetValue(AscendC::GetBlockIdx(), AscendC::GetBlockIdx());
// 算子中多核操作虽然不在同一个地址，但在同一个Cache Line, 会出现数据的随机覆盖，和通用CPU的行为不同
// 调用DataCacheCleanAndInvalid指令后，由于多核操作的时间不一致，最终结果存在随机性，后执行的核会覆盖前面核的结果
AscendC::DataCacheCleanAndInvalid<uint64_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(global);
// 示例4：ENTIRE_DATA_CACHE 模式，假设mmAddr_为0x20（非64B对齐）
// 本样例仅做示例说明，便于开发者理解使用限制，非正常使用样例
AscendC::GlobalTensor<uint64_t> global;
global.SetGlobalBuffer((__gm__ uint64_t*)mmAddr_ + AscendC::GetBlockIdx() * 1024);
for( int i = 0; i < 8; i++) {
   global.SetValue(i, AscendC::GetBlockIdx());
}
// 刷新整个Data Cache，性能较差
AscendC::DataCacheCleanAndInvalid<uint64_t, AscendC::CacheLine::ENTIRE_DATA_CACHE, AscendC::DcciDst::CACHELINE_OUT>(global);
```

