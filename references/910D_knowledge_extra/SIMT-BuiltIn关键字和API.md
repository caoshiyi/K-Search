# SIMT BuiltIn关键字和API<a name="ZH-CN_TOPIC_0000002523311562"></a>

-   \_\_simt\_vf\_\_

    函数标记宏，用于标记SIMT VF入口函数，函数无返回值。使用asc\_vf\_call接口调用SIMT VF入口函数，启动VF子任务。

    ```
    __simt_vf__ inline void KernelAdd(__gm__ float* x, __gm__ float* y, __gm__ float* z)
    ```

    \_\_simt\_vf\_\_标记的SIMT VF函数参数类型支持：

    -   指针类型：\_\_ubuf\_\_ \*、\_\_gm\_\_ \*；
    -   标量类型：bool、int8\_t、uint8\_t、int16\_t、uint16\_t、half、bfloat16、int32\_t、uint32\_t、float、int64\_t、uint64\_t。

-   \_\_simt\_callee\_\_

    函数标记宏，用于标记SIMT VF非入口函数，函数可以有返回值，允许被SIMT VF入口函数或其他非入口函数调用。

    ```
    __simt_callee__ inline float add(float x, float y)
    ```

-   dim3

    内置结构体，用于指定维度信息。dim3的结构体定义为\{dimx，dimy，dimz\}，用于指定3个不同维度的大小，总数为dimx \* dimy \* dimz。开发者可以通过如下方式创建dim3结构。

    ```
    dim3(x); // 创建一维结构，dimy和dimz为默认值1
    dim3(x, y); // 创建二维结构，dimz为默认值1
    dim3(x, y, z); // 创建三维结构
    ```

-   \_\_launch\_bounds\_\_\(N\)

    函数标记宏，在SIMT VF入口函数上可选配置，用于在编译期指定SIMT VF启动的最大线程数。若未配置\_\_launch\_bounds\_\_，最大线程数默认为1024。参数N需要满足：

    -   N \>= dimx \* dimy \* dimz；dimx，dimy，dimz为表示线程的dim3结构体。
    -   N的取值范围为1到2048。

    最大线程数决定了每个线程可分配的寄存器数量，具体对应关系请见下表，寄存器用于存储线程中的局部变量，若局部变量的个数超出寄存器个数，容易出现栈溢出等问题。建议最大线程数与启动VF任务的dim3线程数保持一致。

    **表 1**  \_\_launch\_bounds\_\_的Thread数量与每个Thread可用寄存器数

    <a name="table4946158192213"></a>
    <table><thead align="left"><tr id="row1394585852218"><th class="cellrowborder" valign="top" width="50%" id="mcps1.2.3.1.1"><p id="p13945155822217"><a name="p13945155822217"></a><a name="p13945155822217"></a>Thread的个数(个)</p>
    </th>
    <th class="cellrowborder" valign="top" width="50%" id="mcps1.2.3.1.2"><p id="p1994525802217"><a name="p1994525802217"></a><a name="p1994525802217"></a>每个Thread可用寄存器个数(个)</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="row169451558102213"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p1194575817223"><a name="p1194575817223"></a><a name="p1194575817223"></a>1025~2048</p>
    </td>
    <td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p394545810228"><a name="p394545810228"></a><a name="p394545810228"></a>16</p>
    </td>
    </tr>
    <tr id="row6945205882215"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p1094535892219"><a name="p1094535892219"></a><a name="p1094535892219"></a>513~1024</p>
    </td>
    <td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p8945858152213"><a name="p8945858152213"></a><a name="p8945858152213"></a>32</p>
    </td>
    </tr>
    <tr id="row1394555811221"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p394517586227"><a name="p394517586227"></a><a name="p394517586227"></a>257~512</p>
    </td>
    <td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p2945115817227"><a name="p2945115817227"></a><a name="p2945115817227"></a>64</p>
    </td>
    </tr>
    <tr id="row17946125810227"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p29452058182211"><a name="p29452058182211"></a><a name="p29452058182211"></a>1~256</p>
    </td>
    <td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p1594525822210"><a name="p1594525822210"></a><a name="p1594525822210"></a>127</p>
    </td>
    </tr>
    </tbody>
    </table>

    配置SIMT函数最大线程数为512，示例如下：

    ```
    __simt_vf__ __launch_bounds__(512) inline void add(__gm__ uint8_t* x, __gm__ uint8_t* y, __gm__ uint8_t* z)
    ```

-   blockDim

    内置全局变量，在核函数中可以直接使用，用于获取线程块中配置的线程的三维层次结构，即启动VF时配置的dim3结构体实例值。blockDim.x，blockDim.y，blockDim.z分别表示线程块中三个维度的线程数。

-   gridDim

    内置全局变量，只能在核函数中使用，表示整个计算任务在各个维度上分别由多少个线程块构成。

    -   gridDim.x是x维度上的线程块数量。
    -   gridDim.y是y维度上的线程块数量，目前只能返回1。
    -   gridDim.z是z维度上的线程块数量，目前只能返回1。

-   blockIdx

    内置全局变量，只能在核函数中使用，用于获取块索引。表示当前线程所在的线程块在整个网格中的位置坐标。

    -   blockIdx.x的范围是0到gridDim.x - 1。
    -   blockIdx.y的范围是0到gridDim.y - 1，目前只能返回0。
    -   blockIdx.z的范围是0到gridDim.z - 1，目前只能返回0。

-   threadIdx

    内置全局变量，在核函数中可以直接使用，用于获取当前线程在线程块内部的索引。threadIdx.x，threadIdx.y，threadIdx.z分别表示当前线程在3个维度的索引，threadIdx.x的范围为\[0, blockDim.x\)，threadIdx.y的范围为\[0, blockDim.y\)，threadIdx.z的范围为\[0, blockDim.z\)。线程块内线程的索引与线程ID对应关系如下：

    -   对于一维线程块，其线程ID为threadIdx.x。
    -   对于二维线程块，其线程ID为（threadIdx.x + threadIdx.y \* blockDim.x）。
    -   对于三维线程块，其线程ID为（threadIdx.x + threadIdx.y \* blockDim.x + threadIdx.z \* blockDim.x \* blockDim.y）。

