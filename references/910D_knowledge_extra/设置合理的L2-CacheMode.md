# 设置合理的L2 CacheMode<a name="ZH-CN_TOPIC_0000002523129070"></a>

【优先级】高

> **说明：** 
>该性能优化指导适用于如下产品型号：

【描述】L2 Cache常用于缓存频繁访问的数据，其物理位置如下图所示：

<!-- img2text -->
```text
┌───────────────┐        ┌───────────────┐
│               │        │               │
│    Global     │ ───→   │   L2 Cache    │
│    Memory     │        │               │
│               │ ←───   │               │
└───────────────┘        └───────────────┘
                               │
                    ┌──────────┴──────────┐
                    │                     │
            ┌───────▼───────┐     ┌───────▼───────┐
            │     MTE2      │     │     MTE2      │
            └───────┬───────┘     └───────┬───────┘
                    │                     │
                    ▼                     ▼
             ┌───────────────┐     ┌───────────────┐
             │     Local     │     │     Local     │
             │     Memory    │     │     Memory    │
             └───────┬───────┘     └───────┬───────┘
                     │                     │
          ┌──────────┘                     └──────────┐
          │                                           │
  ┌───────▼───────┐                           ┌───────▼───────┐
  │    Vector     │                           │     Cube      │
  └───────┬───────┘                           └───────┬───────┘
          │                                           │
          └──────────┐                     ┌──────────┘
                     │                     │
             ┌───────▼───────┐     ┌───────▼───────┐
             │     MTE3      │     │    Fixpipe    │
             └───────┬───────┘     └───────┬───────┘
                     │                     │
                     └──────────┬──────────┘
                                ▼
                           ┌───────────────┐
                           │   L2 Cache    │
                           └───────────────┘
```

说明:
- 左侧 `Global Memory` 与 `L2 Cache` 之间为双向访问：
  - `Global Memory → L2 Cache`
  - `L2 Cache → Global Memory`
- 上半部分路径：
  - `L2 Cache ─MTE2→ Local Memory ↔ Vector`
  - `Vector ─MTE3→ L2 Cache`
- 下半部分路径：
  - `L2 Cache ─MTE2→ Local Memory ↔ Cube`
  - `Cube ─Fixpipe→ L2 Cache`

L2 Cache的带宽相比GM的带宽有数倍的提升，因此当数据命中L2 Cache时，数据的搬运耗时会优化数倍。通常情况下，L2 Cache命中率越高，算子的性能越好，在实际访问中需要通过设置合理的L2 CacheMode来保证重复读取的数据尽量缓存在L2 Cache上。

## L2 Cache访问的原理及CacheMode介绍<a name="section75591806575"></a>

数据通过MTE2搬运单元搬入时，L2 Cache访问的典型流程如下：

<!-- img2text -->
```text
            ┌──────┐
            │ 开始 │
            └──┬───┘
               │
               ▼
        ┌───────────────┐
        │  L2 Cache     │
        │      命中     │
        └──────┬────────┘
               │
         ┌─────┴─────┐
         │           │
        是           否
         │           │
         ▼           ▼
┌────────────────┐   ┌────────────────┐
│  L2 Cache ->   │   │  分配CacheLine │
│  Local Memory  │   └────────┬───────┘
└────────┬───────┘            │
         │                    ▼
         │            ┌───────────────┐
         │            │  cacheline    │
         │            │     状态      │
         │            └──────┬────────┘
         │                   │
         │             ┌─────┴─────┐
         │             │           │
         │          dirty       clean
         │             │           │
         │             ▼           ▼
         │   ┌────────────────┐  ┌────────────────┐
         │   │   旧数据       │  │   GM -> L2     │
         │   │ L2 Cache -> GM │  │    Cache       │
         │   └────────┬───────┘  └────────┬───────┘
         │            │                   │
         │            └────────┬──────────┘
         │                     │
         └─────────────────────◄
                               │
                               ▼
                           ┌──────┐
                           │ 结束 │
                           └──────┘
```

数据通过MTE3或者Fixpipe搬运单元搬出时，L2 Cache访问的典型流程如下：

<!-- img2text -->
```text
            ┌──────┐
            │ 开始 │
            └──┬───┘
               │
               ▼
        /────────────────\
       /   L2 Cache命中   \
       \                  /
        \────────────────/
           │是        │否
           ▼          ▼
┌──────────────────┐   ┌────────────────┐
│  Local Memory    │   │  分配CacheLine │
│   ->L2 Cache     │   └────────┬───────┘
└────────┬─────────┘            │
         │                      ▼
         │              /────────────────\
         │             /  CacheLine状态   \
         │             \                  /
         │              \────────────────/
         │                │dirty     │clean
         │                ▼          │
         │     ┌──────────────────┐  │
         │     │      旧数据       │  │
         │     │ L2 Cache -> GM    │  │
         │     └─────────┬────────┘  │
         │               │           │
         └───────────────┴───────────┘
                         │
                         ▼
           ┌────────────────────────┐
           │ CacheLine状态置为dirty │
           └───────────┬────────────┘
                       │
                       ▼
                    ┌──────┐
                    │ 结束 │
                    └──────┘
```

从上面的流程可以看出，当数据访问总量超出L2 Cache容量时，AI Core会对L2 Cache进行数据替换。由于Cache一致性的要求，替换过程中旧数据需要先写回GM（此过程中会占用GM带宽），旧数据写回后，新的数据才能进入L2 Cache。

开发者可以针对访问的数据设置其CacheMode，对于只访问一次的Global Memory数据设置其访问状态为不进入L2 Cache，这样可以更加高效的利用L2 Cache缓存需要重复读取的数据，避免一次性访问的数据替换有效数据。

## 设置L2 CacheMode的方法<a name="section058817270210"></a>

Ascend C基于GlobalTensor提供了SetL2CacheHint接口，用户可以根据需要指定CacheMode。

考虑如下场景，构造两个Tensor的计算，x的输入Shape为\(5120, 5120\)，y的输入Shape为\(5120, 15360\)，z的输出Shape为\(5120, 15360\)，由于两个Tensor的Shape不相等，x分别与y的3个数据块依次相加。该方案主要为了演示CacheMode的功能，示例代码中故意使用重复搬运x的实现方式，真实设计中并不需要采用这个方案。下文完整样例请参考[设置合理L2 CacheMode样例](https://gitee.com/ascend/samples/tree/master/operator/ascendc/4_best_practices/12_l2_cache_bypass)。

<!-- img2text -->
```
┌──────────────────────────────┐  ┌──────────┐  ┌──────────────────────────────┐
│              z               │  │    x     │  │              y               │
│           │        │         │  │          │  │           │        │         │
└──────────────────────────────┘  └──────────┘  └──────────────────────────────┘
```

<a name="table12921549195512"></a>
<table><thead align="left"><tr id="row1229364945511"><th class="cellrowborder" valign="top" width="6.813978389954251%" id="mcps1.1.4.1.1"><p id="p2081249145715"><a name="p2081249145715"></a><a name="p2081249145715"></a>实现方案</p>
</th>
<th class="cellrowborder" valign="top" width="42.67497323079919%" id="mcps1.1.4.1.2"><p id="p2029374985519"><a name="p2029374985519"></a><a name="p2029374985519"></a>原始实现</p>
</th>
<th class="cellrowborder" valign="top" width="50.51104837924658%" id="mcps1.1.4.1.3"><p id="p152931149115516"><a name="p152931149115516"></a><a name="p152931149115516"></a>优化实现</p>
</th>
</tr>
</thead>
<tbody><tr id="row1629374995517"><td class="cellrowborder" valign="top" width="6.813978389954251%" headers="mcps1.1.4.1.1 "><p id="p108124995710"><a name="p108124995710"></a><a name="p108124995710"></a>实现方法</p>
</td>
<td class="cellrowborder" valign="top" width="42.67497323079919%" headers="mcps1.1.4.1.2 "><p id="p1764591574211"><a name="p1764591574211"></a><a name="p1764591574211"></a>总数据量700MB，其中：x：100MB；y：300MB；z：300MB。</p>
<p id="p13628115612406"><a name="p13628115612406"></a><a name="p13628115612406"></a>使用40个核参与计算，按列方向切分。</p>
<p id="p5900191641514"><a name="p5900191641514"></a><a name="p5900191641514"></a>x、y、z 对应GlobalTensor的CacheMode均设置为CACHE_MODE_NORMAL，需要经过L2 Cache，需要进入L2 Cache的总数据量为700MB。</p>
</td>
<td class="cellrowborder" valign="top" width="50.51104837924658%" headers="mcps1.1.4.1.3 "><p id="p2198131313619"><a name="p2198131313619"></a><a name="p2198131313619"></a>总数据量700MB，其中：x：100MB；y：300MB；z：300MB。</p>
<p id="p3198111373618"><a name="p3198111373618"></a><a name="p3198111373618"></a>使用40个核参与计算，按列方向切分。</p>
<p id="p1545392118509"><a name="p1545392118509"></a><a name="p1545392118509"></a>x对应的GlobalTensor的CacheMode设置为CACHE_MODE_NORMAL；y和z对应的GlobalTensor的CacheMode设置为CACHE_MODE_DISABLE。只有需要频繁访问的x，设置为需要经过L2 Cache。需要进入L2 Cache的总数据量为100MB。</p>
</td>
</tr>
<tr id="row3293124918559"><td class="cellrowborder" valign="top" width="6.813978389954251%" headers="mcps1.1.4.1.1 "><p id="p12812993573"><a name="p12812993573"></a><a name="p12812993573"></a>示例代码</p>
</td>
<td class="cellrowborder" valign="top" width="42.67497323079919%" headers="mcps1.1.4.1.2 "><a name="screen924835613570"></a><a name="screen924835613570"></a><pre class="screen" codetype="Cpp" id="screen924835613570">xGm.SetGlobalBuffer((__gm__ float *)x + AscendC::GetBlockIdx() * TILE_N);
yGm.SetGlobalBuffer((__gm__ float *)y + AscendC::GetBlockIdx() * TILE_N);
zGm.SetGlobalBuffer((__gm__ float *)z + AscendC::GetBlockIdx() * TILE_N);</pre>
</td>
<td class="cellrowborder" valign="top" width="50.51104837924658%" headers="mcps1.1.4.1.3 "><a name="screen271414925813"></a><a name="screen271414925813"></a><pre class="screen" codetype="Cpp" id="screen271414925813">xGm.SetGlobalBuffer((__gm__ float *)x + AscendC::GetBlockIdx() * TILE_N);
yGm.SetGlobalBuffer((__gm__ float *)y + AscendC::GetBlockIdx() * TILE_N);
zGm.SetGlobalBuffer((__gm__ float *)z + AscendC::GetBlockIdx() * TILE_N);
// disable the L2 cache mode of y and z
yGm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
zGm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);</pre>
</td>
</tr>
</tbody>
</table>

> **说明：** 
>你可以通过执行如下命令行，通过msprof工具获取上述示例的性能数据并进行对比。
>```
>msprof op --launch-count=2 --output=./prof ./execute_add_op
>```
>重点关注Memory.csv中的aiv\_gm\_to\_ub\_bw\(GB/s\)和aiv\_main\_mem\_write\_bw\(GB/s\)写带宽的速率。

