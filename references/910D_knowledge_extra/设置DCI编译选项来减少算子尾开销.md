# 设置DCI编译选项来减少算子尾开销<a name="ZH-CN_TOPIC_0000002554409037"></a>

> **说明：** 
>该性能优化建议适用于如下型号：
>-   Ascend 950PR/Ascend 950DT

【优先级】高

【描述】算子执行结束时，需要将DCache置为无效，防止后续算子继续使用DCache中的数据而受到影响。可以通过在编译选项中添加--cce-no-dcache-flush=true，用于在算子尾部增加DCI（DataCacheInvalid）指令来使DCache失效。如果不开启该选项，则会默认增加DCCI（DataCacheCleanAndInvalid）指令来使DCache失效。

插入DCI指令相比于插入DCCI指令，其减少了数据从DCache同步到GM（Clean）的过程，性能上会有一定优势。插入DCCI是一种额外的容错保证，如果开发者使用了\* \_\_gm\_\_的方式改写GM内存，或者调用GlobalTensor.SetValue函数时，没有正确的调用DataCacheCleanAndInvalid接口来保证Cache一致性，编译框架自动插入DCCI恰好可以保证算子精度正常。

所以在如下场景，可以通过开启该编译选项来降低算子尾部开销：

-   算子使用\* \_\_gm\_\_的方式改写GM内存，或者调用GlobalTensor.SetValue函数时，正确的使用DataCacheCleanAndInvalid接口，手动将数据从DCache中回刷到GM上，保证Cache的一致性。不依赖编译框架自动插入DCCI指令来保证一致性。
-   算子不包含使用\* \_\_gm\_\_的方式改写GM内存，或者调用GlobalTensor.SetValue函数的代码。

