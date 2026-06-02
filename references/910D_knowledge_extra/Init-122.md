# Init<a name="ZH-CN_TOPIC_0000002554343527"></a>

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
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p1695957133611"><a name="p1695957133611"></a><a name="p1695957133611"></a>x</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section4299173755913"></a>

Init主要用于对Conv3D对象中的Tiling数据进行初始化，根据Tiling参数进行资源划分，同时获取用户声明的Pipe对象，完成内存分配。Tiling参数的具体介绍请参考[Conv3D Tiling](Conv3D-Tiling侧接口.md)。

## 函数原型<a name="section079519516019"></a>

```
__aicore__ inline void Init(const void* __restrict cubeTiling)
```

## 参数说明<a name="section423183813019"></a>

<a name="table17247917193819"></a>
<table><thead align="left"><tr id="row826411177387"><th class="cellrowborder" valign="top" width="21.43%" id="mcps1.1.4.1.1"><p id="p1626431733817"><a name="p1626431733817"></a><a name="p1626431733817"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="19.39%" id="mcps1.1.4.1.2"><p id="p526411177386"><a name="p526411177386"></a><a name="p526411177386"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="59.18%" id="mcps1.1.4.1.3"><p id="p122641917173815"><a name="p122641917173815"></a><a name="p122641917173815"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row1526491719388"><td class="cellrowborder" valign="top" width="21.43%" headers="mcps1.1.4.1.1 "><p id="p17264191712384"><a name="p17264191712384"></a><a name="p17264191712384"></a>cubeTiling</p>
</td>
<td class="cellrowborder" valign="top" width="19.39%" headers="mcps1.1.4.1.2 "><p id="p112644172389"><a name="p112644172389"></a><a name="p112644172389"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="59.18%" headers="mcps1.1.4.1.3 "><p id="p88648241418"><a name="p88648241418"></a><a name="p88648241418"></a>Conv3D对象的Tiling参数，Tiling结构体定义请参见<a href="TConv3DApiTiling结构体.md">TConv3DApiTiling结构体</a>。</p>
<p id="p5264151723816"><a name="p5264151723816"></a><a name="p5264151723816"></a>Tiling参数可以通过Host侧<a href="GetTiling-126.md">GetTiling</a>接口获取，并传递到Kernel侧使用。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section11828111017"></a>

无

## 约束说明<a name="section29522578115"></a>

-   调用Init接口前必须先初始化TPipe。
-   Init接口必须在IterateAll和End接口前调用，且只能调用一次Init接口，调用顺序如下。

    ```
    Init(...);
    ...
    IterateAll(...);
    End();
    ```

## 调用示例<a name="section1242919111927"></a>

```
TPipe pipe;
conv3dApi.Init(&tiling);
```

