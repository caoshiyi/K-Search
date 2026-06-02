# TilingData结构体<a name="ZH-CN_TOPIC_0000002523344188"></a>

## 功能说明<a name="section77005157231"></a>

在算子实现中，由Tiling组装通信配置项，核函数实现时将Tiling配置项通过入参传递给Kernel侧通信API做通信计算。本节TilingData结构体包括[Mc2InitTiling](#table4835205712588)和[Mc2CcTiling](#table678914014562)，这两个结构体均通过调用[GetTiling](GetTiling-121.md)接口返回。其中，[Mc2CcTiling](#table678914014562)为具体每个通信任务的参数配置，当算子中有多个通信任务时，可定义多个[Mc2CcTiling](#table678914014562)参数（最多支持定义8个）。

## 参数说明<a name="section11721925114810"></a>

**表 1**  Mc2InitTiling参数说明

<a name="table4835205712588"></a>
<table><thead align="left"><tr id="row118356578583"><th class="cellrowborder" valign="top" width="21.94%" id="mcps1.2.3.1.1"><p id="p48354572582"><a name="p48354572582"></a><a name="p48354572582"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="78.06%" id="mcps1.2.3.1.2"><p id="p583535795817"><a name="p583535795817"></a><a name="p583535795817"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row18835145716587"><td class="cellrowborder" valign="top" width="21.94%" headers="mcps1.2.3.1.1 "><p id="p62941751537"><a name="p62941751537"></a><a name="p62941751537"></a>reserved</p>
</td>
<td class="cellrowborder" valign="top" width="78.06%" headers="mcps1.2.3.1.2 "><p id="p179841930161220"><a name="p179841930161220"></a><a name="p179841930161220"></a>初始化通信任务配置。uint8_t *类型，支持最大长度64字节，该结构体仅支持通过接口<a href="GetTiling-121.md">GetTiling</a>获取。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  Mc2CcTiling参数说明

<a name="table678914014562"></a>
<table><thead align="left"><tr id="row878919012561"><th class="cellrowborder" valign="top" width="21.72%" id="mcps1.2.3.1.1"><p id="p1778912095611"><a name="p1778912095611"></a><a name="p1778912095611"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="78.28%" id="mcps1.2.3.1.2"><p id="p27891055616"><a name="p27891055616"></a><a name="p27891055616"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row07893011565"><td class="cellrowborder" valign="top" width="21.72%" headers="mcps1.2.3.1.1 "><p id="p878910085614"><a name="p878910085614"></a><a name="p878910085614"></a>reserved</p>
</td>
<td class="cellrowborder" valign="top" width="78.28%" headers="mcps1.2.3.1.2 "><p id="p1678950205618"><a name="p1678950205618"></a><a name="p1678950205618"></a>各通信域中每个通信任务的参数配置。uint8_t *类型，支持最大长度280字节，该结构体仅支持通过接口<a href="GetTiling-121.md">GetTiling</a>获取。注意，最多支持配置8个通信任务。</p>
</td>
</tr>
</tbody>
</table>

