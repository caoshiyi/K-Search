# fork方式创建子进程导致应用进程卡死

**页面ID:** troubleshooting_0075  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/maintenref/troubleshooting/troubleshooting_0075.html

---

#### 问题现象

多卡训练场景下，出现训练进程卡死，或者出现训练超时、用户dataloader卡死等现象。

#### 原因分析

1. 查看Python堆栈，堆栈信息中包含**fork**关键字。

使用pyspy工具查看Python堆栈信息的命令示例如下。

使用pyspy命令前，需要安装gdb和py-spy。若环境中未安装gdb，可通过包管理（如**apt-get install gd****b**命令、**yum install gdb**命令）进行安装，详细安装步骤及使用方法请参见[GDB官方文档](https://sourceware.org/gdb/)；若环境中未安装py-spy，可使用**pip3 install py-spy**命令安装（若安装时提示pip版本低，例如*You are using pip version 19.2.3, however version 24.0 is available*，这时可按照提示使用pip3 install --upgrade pip命令升级pip即可）。

```
# 将指定进程的堆栈信息导出到指定文件中，*pid*表示卡住的用户进程ID，*pyspy.log*表示存放堆栈信息的文件，请根据实际情况替换
py-spy dump -p *pid > **pyspy.log*
```

堆栈信息示例如下（*xxxx*表示目录名称、*trainApp*表示训练程序，由实际业务情况决定，此处仅为示例）：

```
1 Process 16203: /train/*xxxx*/*xxxx*/*xxxx*/python3.8 -u -m *trainApp* --config-dir
2 Python  v3.8.19 (/train/*xxxx*/*xxxx*/*xxxx*/python3.8)
3
4 Thread 0xFFFF9CF35B50 (active): "MainThread"
5    poll (multiprocessing/**popen_fork.py**:27)
6    wait (multiprocessing/**popen_fork.py**:47)
7    join (multiprocessing/process.py:149)
8    _terminate_pool (multiprocessing/pool.py:729)
9    __call__ (multiprocessing/util.py:224)
10   _scale_down_hw (datasets/datasets.py:96)
11   __init__ (datasets/datasets.py:73)
......
```

2. 查看C/C++堆栈，堆栈信息中包含**acquire_lock**关键字。

       通过gdb命令观察卡住进程的调用栈信息，若环境中未安装gdb，则需要安装gdb，可通过包管理（如apt-get install gdb、yum install gdb）进行安装，详细安装步骤及使用方法请参见[GDB官方文档](https://sourceware.org/gdb/)。

```
# 先执行gdb命令,*pid*表示卡住的用户进程ID，请根据实际情况替换
gdb -p *pid*
# 再查看调用栈
(gdb)bt
```

堆栈信息示例如下：

```
#0 0x0000ffffa9b2b268 in do_futex_wait.constprop () from /lib/aarch64-linux-gnu/libpthread.so.0
#1 0x0000ffffa9b2b39c in   new_sem_waut_slow.constprop.0 () from /lib/aarch64-linux-gnu/libpthread.so.0
#2 0x0000ffffa9e96eb8 in PyThread_**acquire_lock**_timed () from /usr/local/lib/libpython3.8.so.1.0
#3 0x0000ffffa9e865a8 in _PyThreadState_DeleteExcept () from /usr/local/lib/libpython3.8.so.1.0
#4 0x0000ffffa9eb94ac in _PyOS_AfterFork_Child () from /usr/local/lib/libpython3.8.so.1.0
#5 0x0000ffffa9eb9638 in ?? () from /usr/local/lib/libpython3.8.so.1.0
......
```

3. 通过Python堆栈的**fork**关键字以及C++堆栈的**acquire_lock**关键字，确认训练进程卡死是因为用fork方式启进程触发Python的bug而导致的问题。

在Python3.8~Python3.11版本中如果不指定创建进程的方式，或者显式指定为fork时，在创建子进程时可能会复制主进程的锁状态，而在子进程里再触发获取锁时，就会导致死锁，进而导致业务进程卡死。

Python社区也有相关说明：python社区有相同问题的issue：[https://github.com/python/cpython/issues/74580](https://github.com/python/cpython/issues/74580)

#### 解决方法

两种解决方式，由用户根据业务情况选用：

- **方式一：按照[Python官网](https://www.python.org/downloads/)的指导，升级Python3.8~Python3.11版本的补丁。**

在Python官网，针对Python3.8~Python3.11版本都出了补丁版本，解决fork方式引起的bug。

<!-- img2text -->
```
Looking for a specific release?
Python releases by version number:

┌──────────────────────┬────────────────┬──────────────────┬────────────────┐
│ Release version      │ Release date   │                  │ Click for more │
├──────────────────────┼────────────────┼──────────────────┼────────────────┤
│ Python 3.11.10       │ Sept. 7, 2024  │ Download         │ Release Notes  │
│ Python 3.10.15       │ Sept. 7, 2024  │ Download         │ Release Notes  │
│ Python 3.12.6        │ Sept. 6, 2024  │ Download         │ Release Notes  │
│ Python 3.9.20        │ Sept. 6, 2024  │ Download         │ Release Notes  │
│ Python 3.8.20        │ Sept. 6, 2024  │ Download         │ Release Notes  │
├──────────────────────┼────────────────┼──────────────────┼────────────────┤
│ Python 3.12.5        │ Aug. 6, 2024   │ Download         │ Release Notes  │
│ Python 3.12.4        │ June 6, 2024   │ Download         │ Release Notes  │
└──────────────────────┴────────────────┴──────────────────┴────────────────┘

View older releases
```

在这些补丁版本中，也有针对fork问题的相应说明，如下：

<!-- img2text -->
```
Core and Builtins

• gh-112275: A deadlock involving pystate.c’s HEAD_LOCK in posixmodule.c at fork is
  now fixed. Patch by ChuBoning based on previous Python 3.12 fix by Victor Stinner.
```

- **方式二：修改客户业务代码，显式使用forkserver或者spawn方式。**
       注意事项：如果涉及修改fork的地方比较多或工作量比较大，建议采用方式一，防止修改遗漏。

  1. 找到Python安装目录

执行**pip show torch**命令查找Python安装目录，查询结果示例如下：

<!-- img2text -->
```
pip show torch

Name: torch
Version: 2.1.0
Summary: Tensors and Dynamic neural networks in Python with strong GPU acceleration
Home-page: https://pytorch.org/
Author: PyTorch Team
Author-email: packages@pytorch.org
License: BSD-3
Location: /root/anaconda3/envs/lw38/lib/python3.8/site-packages
Requires: filelock, fsspec, jinja2, networkx, sympy, typing-extensions
Required-by: torch-npu
```

  2. 在Python安装目录下执行**find -name popen_fork.py**命令找到popen_fork.py文件，在fork启进程的地方都增加触发堆栈的代码。

在_launch(self,process_obj)函数内添加代码，目的是走fork的子进程都触发堆栈：

<!-- img2text -->
```
popen_fork.py

62  def kill(self):
63      self._send_signal(signal.SIGKILL)

66  def _launch(self, process_obj):
67      code = 1
68      parent_r, child_w = os.pipe()
69      child_r, parent_w = os.pipe()
70
71      self.pid = os.fork()
72      if self.pid == 0:
73          try:
74              os.close(parent_r)
75              os.close(parent_w)
76              code = process_obj._bootstrap(parent_sentinel=child_r)
77          finally:
78              os._exit(code)
```

说明:
- 红框标注位置覆盖第70-71行附近，重点指向 `self.pid = os.fork()` 插入/定位处。
- 上下文含义：在 `_launch(self, process_obj)` 函数内，于 fork 启进程的位置触发堆栈打印。

例如，在上图70行的位置增加如下代码，用于在fork启进程的地方触发堆栈消息打印：

```
import traceback
import  time
timestamp = time.time()
timestamp_str = str(int(timestamp))
file_name = f"stack_{timestamp_str}.txt"
with open("/home/{}.txt".format(file_name),"a") as f:
    traceback.print_stack(file=f)
```

  3. 复跑训练业务，再次查看Python堆栈。

在堆栈信息中，先忽略CANN相关的堆栈，只要修改客户业务的fork，切换为“spawn”或“forkserver”启动方式。关于启动方式的详细使用说明，请参见[Python官网文档](https://docs.python.org/zh-cn/3.8/library/multiprocessing.html)（此处可根据所用的Python版本，选择对应版本的文档）。
