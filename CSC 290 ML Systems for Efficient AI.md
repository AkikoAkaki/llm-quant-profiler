---
date created: 2026-05-04
date updated: 2026-05-14
---

# Part I：计算机组织基础（Computer Organization）

> 基于 Sreepathi Pai，University of Rochester，Fall 2025 课程材料整理
> 覆盖 Module 1–5，对应 Aug 25 – Sep 29 课程内容

## Module 1：课程导论与效率模型

### 1.1 什么是 ML 系统

ML 系统（ML System）是指任何能运行 ML 程序的计算机系统。课程使用两个维度来分类：

**按内存拓扑分类：**

|类型|含义|例子|
|---|---|---|
|**Shared Memory System**|单一内存空间（通常指单台机器）|一台配有多 GPU 的服务器|
|**Distributed Memory System**|至少两台机器通过网络连接；读取对方内存需要显式数据传输|GPU 集群、数据中心|

**ML 程序的三种执行模态：**

- **Training（训练）**：从随机权重出发，通过前向传播计算损失、反向传播更新权重，直至收敛。计算量最大，时间最长，几乎仅限于大型机构。
- **Fine-tuning（微调）**：以预训练权重为起点，针对特定任务低成本再训练。每个下游任务做一次。
- **Inference（推理）**：仅做前向传播。每次用户请求触发一次。计算量最小。

**ML 程序的三个核心特征：**

- 大量 **Compute**（算术运算量）
- 大量 **Memory**（数据规模）
- 大量 **Communication**（分布式系统间的数据传输）

### 1.2 效率（Efficiency）的定义框架

> "用最少的资源完成有用的工作。"

这门课关注的不是算法层面的 $O(n)$ vs $O(n^2)$，而是**给定算法在硬件上的实际执行效率**。

#### 时间模型

$$T = \frac{W \times t}{P}$$

其中：

- $W$：工作量（Work，例如 FLOP 数）
- $t$：单位工作的平均时间
- $P$：并行度（Parallelism）

**示例**：煮咖啡需要 10 分钟，煎蛋需要 5 分钟。单炉灶串行需要 15 分钟；双炉灶并行只需要 10 分钟（受限于较长的那个任务）。

#### 能耗模型

$$E = \sum_{w \in W} E_w$$

其中 $E_w$ 是执行工作 $w$ 消耗的能量。测量方式包括外部传感器和处理器内部传感器（如 Intel 的 **RAPL**，Read Energy, Power, and Temperature）。

#### 存储效率（数据压缩）

$$\text{Compression Ratio} = \frac{\text{uncompressed size}}{\text{compressed size}}$$

- **无损压缩（Lossless）**：可以完全恢复原始数据
- **有损压缩（Lossy）**：无法精确恢复，但压缩后的数据在应用层面足够接近原始数据（ML 中的量化本质上是有损压缩）

#### 可扩展性（Scalability）

描述资源需求随工作量增长的变化方式，与 **Amdahl's Law**（后续章节）密切相关。

### 1.3 课程三阶段总览

|阶段|内容|核心问题|
|---|---|---|
|**Part I**：Computer Organization|CPU/GPU 架构、数据表示、内存系统|程序在硬件上是怎么跑的？|
|**Part II**：Real-World Efficiency|ML 计算图、Loop 优化、Roofline、训练系统|ML 程序有哪些效率瓶颈，如何建模和优化？|
|**Part III**：Advanced Topics|专用加速器、量化、分布式推理、Scaling Laws|前沿研究在做什么？|

---

## Module 2：Compute — CPUs

### 2.1 von Neumann 架构基础

#### 基本组成

```
┌─────────────────────────┐
│           CPU           │
│  ┌──────┐  ┌─────────┐  │
│  │ ALU  │  │   MEM   │  │
│  └──────┘  └─────────┘  │
│       ┌──────────┐      │
│       │  Regs    │      │
│       │  (PC...) │      │
│       └──────────┘      │
└────────────┬────────────┘
             │
      ┌──────┴──────┐
      │     RAM     │
      │  (Program   │
      │   + Data)   │
      └─────────────┘
```

程序和数据都存储在内存中，这是 von Neumann 架构的核心思想（也是后来"代码即数据"思想的基础）。

#### CPU 指令执行周期

CPU 每条指令的执行过程分为以下步骤：

1. **FETCH**：从内存中读取指令（程序计数器 PC 指向当前指令地址）
2. **DECODE**：解析指令操作码（Opcode）和操作数（Operands）
3. **EXECUTE**：在功能单元中执行运算
4. **WRITE BACK**：将结果写回寄存器或内存，并使结果对后续指令可见

每个步骤由**时钟（Clock）** 协调，一个或多个时钟周期完成一步。

#### 寄存器（Registers）

- CPU 内部速度**最快**的存储形式（顺序逻辑）
- 按名称访问（如 `EAX`、`RBX`、`R10`），数量少（现代 CPU 几十个）
- 每个寄存器存储 32 或 64 位数据
- **特殊寄存器**：程序计数器（PC），始终包含当前执行指令的地址

#### 指令类型

```
指令类型            示例助记符
──────────────────────────────
整型算术            IADD, ISUB, IMUL
浮点算术            FADD, FMUL, FDIV
比较                GT, LT, GTE
逻辑/位运算         AND, OR, XOR, SHR
内存访问            LD (load), ST (store)
控制流              JMP (无条件), JC (条件跳转)
```

> **注意**：这里的助记符是教学用的通用示例，不对应任何特定 ISA。

#### ISA（指令集架构）

ISA 是程序员与处理器的接口契约，定义了处理器"理解"哪些指令。同一 ISA 可以由不同 **microarchitecture（微架构）** 实现：

| ISA    | 代表实现                  | 类型          |
| ------ | --------------------- | ----------- |
| x86-64 | Intel Core, AMD Ryzen | CISC（复杂指令集） |
| ARMv8  | Apple M 系列, Qualcomm  | RISC（精简指令集） |
| RISC-V | 开源                    | RISC        |

> **CISC vs RISC** 的历史区分（复杂指令集 vs 精简指令集）已逐渐淡化。现代 CISC 处理器（如 x86）内部会将复杂指令拆解为类似 RISC 的微操作（μops）执行。

#### 功能单元（Functional Units）

- **ALU**（Arithmetic Logic Unit）：执行整数算术和逻辑运算
- **FPU**（Floating Point Unit）：执行浮点运算
- **MEM Unit**：执行 Load/Store
- **MMA / Tensor Core**：矩阵乘法加速单元（用于 ML 的专用硬件）

---

### 2.2 流水线执行（Pipelined Execution）

#### 串行执行的浪费

在朴素的串行执行中，每条指令必须完全执行完毕才能开始下一条：

```
时间轴 →
指令1: [F][D][E][WB]
指令2:             [F][D][E][WB]
指令3:                         [F][D][E][WB]
```

大量硬件处于空闲状态，效率极低。

#### 流水线执行

**流水线（Pipeline）** 将指令执行分成多个阶段，多条指令同时处于不同阶段：

```
时间轴 →
指令1: [F][D][E][WB]
指令2:    [F][D][E ][WB]
指令3:       [F][D ][E][WB]
指令4:          [F ][D][E][WB]
```

稳定状态下，每个时钟周期完成一条指令（IPC = 1）。这是流水线的理论峰值。

#### 流水线中的三类冒险（Hazards）

真实程序中，流水线无法总保持满负荷，原因是存在三类冒险：

**① 数据冒险（Data Hazard）—— 写后读（RAW, Read After Write）**

```asm
ADD R3, R1, R2    ; 将 R3 写入
SUB R4, R3, 1    ; 需要读取 R3 ← 但 R3 还没写完！
```

SUB 指令在 EXECUTE 阶段需要读 R3，而 ADD 还未到 WRITE BACK 阶段。解决方案是**停顿（Stall）**：暂停后续指令直到前一指令完成写回，或使用**数据前递（Data Forwarding / Bypassing）**将运算结果直接传给下一条指令，而无需等待写回内存。

**② 结构冒险（Structural Hazard）—— 硬件资源竞争**

```asm
DIV R5, R1, R2   ; 除法需要多个周期
DIV R6, R3, R4   ; 第二条除法想用同一个除法器
```

某些功能单元（如除法器）无法同时被多条指令使用。解决方案是停顿、增加功能单元副本、或流水化功能单元。

**③ 控制冒险（Control Hazard）—— 分支跳转**

```asm
CMP  R1, R2, R3
JL   addr1        ; 条件跳转：往哪里取指？
SUB  R3, R3, 1    ; 如果不跳转
...
addr1:
MUL  R4, R2, R3   ; 如果跳转
```

在 JL 被 FETCH 和 DECODE 后，CPU 不知道下一条指令应该取 SUB 还是 MUL，必须等 CMP 执行完毕才能确定。解决方案是停顿，或使用**分支预测（Branch Prediction）**进行投机执行。

#### IPC（每周期指令数）

IPC = Instructions Per Cycle，是衡量处理器利用率的核心指标。

- 经典五段流水线的**理论 IPC = 1**
- 实际 IPC 因各类冒险导致的停顿而降低
- **超标量**处理器的理论 IPC > 1

---

### 2.3 乱序执行（Out-of-Order Execution）

#### 超标量执行（Superscalar Execution）

现代处理器不满足于 IPC = 1，通过**同时 Fetch、Decode、Execute 多条指令**来提高 IPC：

```
时间轴（超标量，每周期取2条）→
[F:I1,I2][D:I1,I2][E:I1,I2][WB:I1,I2]
         [F:I3,I4][D:I3,I4][E:I3,I4][WB:I3,I4]
```

但同时执行多条指令会引入新的相关性问题：

**反相关（Anti-dependence / WAR, Write After Read）：**

```asm
ADD R1, R2, R3    ; 读 R2
MUL R2, R4, R5   ; 写 R2 ← 但 ADD 可能还没读完
```

ADD 和 MUL 在逻辑上是独立的，但 MUL 写 R2 而 ADD 还要读 R2。若顺序执行无问题，但同时执行会产生错误。

**输出相关（Output dependence / WAW, Write After Write）：**

```asm
ADD R1, R2, R3    ; 写 R1
SUB R1, R4, R5   ; 也写 R1 ← 谁先写谁后写？
```

#### 寄存器重命名（Register Renaming）

这是解决上述假相关（false dependences）的核心技术：

**ISA 寄存器（逻辑寄存器）** 是程序员看到的有限寄存器名（如 R1, R2, ...）。**物理寄存器（Physical Registers）** 是处理器内部实际拥有的更多寄存器池。

处理器在执行时动态地将逻辑寄存器**映射**到不同的物理寄存器，从而消除 WAR 和 WAW 假相关，只保留真正的数据依赖（RAW）。

#### 指令窗口 + 数据流执行 = 乱序执行

**指令窗口（Instruction Window）** 是处理器在内部缓存的一批已解码但还未执行的指令。处理器在这批指令中寻找**相互独立**（无 RAW 依赖）的指令，按数据流驱动顺序（而非程序顺序）执行：

```asm
add r3, r1, r2      ; 依赖 r1, r2 → 先执行
sub r5, r3, 1       ; 依赖 r3 → 等 add 完
add r4, r1, 3       ; 独立 → 与 sub 并行执行
mul r6, r3, r4      ; 依赖 r3 和 r4 → 等两者都完成
shr r7, r5, 2       ; 依赖 r5 → 等 sub 完
```

数据流图（DAG）：

```
    add(r1,r2)→r3
       /        \
  sub(r3)→r5  add(r1,3)→r4
      |               |
  shr(r5)→r7    mul(r3,r4)→r6
```

指令的执行顺序由依赖关系决定，而非程序文本顺序。

#### 投机执行（Speculative Execution）

条件分支会截断指令窗口（无法跨越分支 Fetch 指令）。**分支预测**机制通过预测分支方向来扩大有效指令窗口：

1. 处理器**预测**分支方向，开始 Fetch 并执行预测路径上的指令
2. 当条件分支的实际结果确定后，检验预测是否正确：
    - **预测正确**：继续，这些指令的结果合法提交，节省了时间
    - **预测错误**：**刷新流水线（Pipeline Flush）**，丢弃所有已投机执行的指令，从正确路径重新 Fetch

现代处理器的分支预测准确率通常在 95%–99% 以上，预测错误代价随流水线深度增加（Intel 的深流水线错误代价可达 15–20 个周期）。

#### 乱序执行的规模

- 现代处理器每次 Fetch 8–16 条指令
- 稳定状态下，数百条指令同时处于"飞行中"（in-flight）
- 指令窗口受限于条件分支、寄存器物理数量等

---

### 2.4 现代多核与性能上限

#### 同时多线程（SMT / Hyperthreading）

不同线程的指令天然独立（没有数据依赖）。SMT 通过将多个线程的指令**同时送入同一超标量后端**来充分利用硬件资源：

```
线程A的前端  ──┐
               ├──→ 共享的乱序超标量后端（ALU、FPU等）
线程B的前端  ──┘
```

每个线程有独立的 PC 和寄存器状态（前端），但共享 ALU、Cache 等后端资源。Intel 称为 Hyperthreading，每个物理核通常呈现为 2 个逻辑核。

> SMT 并不总是有益。若两个线程竞争相同的后端资源（例如都是浮点密集型），反而会相互干扰。

#### 芯片多处理器（CMP / Multicore）

**多核**处理器在单块芯片上复制多个完整的处理器核心，共享同一内存：

```
┌──────────────────────────────┐
│  Core 0  │  Core 1  │  ...  │
│  L1      │  L1      │       │
├──────────┴──────────┴───────┤
│          共享 L3 Cache       │
├─────────────────────────────┤
│          Memory Controller  │
└─────────────────────────────┘
```

每个核心可以运行独立程序，也可以运行同一程序的不同线程（需要程序使用 pthreads、OpenMP 等并发编程接口）。

#### 程序性能上限公式

给定一个处理器，程序执行时间的理论下界（以时钟周期计）为：

$$T_{min\text{ (cycles)}} = \frac{W}{IPC_{max}}$$

将其换算为时间（秒）：

$$T_{min\text{ (seconds)}} = \frac{W}{IPC_{max} \times f}$$

其中 $f$ 是处理器时钟频率（Hz）。

**实践意义**：测量程序实际达到的 IPC，与 $IPC_{max}$ 对比，差距说明了优化空间的大小。注意测量的是**有用 IPC**（不包含因 Stall 浪费的周期）。

#### "喂饱"现代 CPU 的条件

充分利用现代 CPU 需要同时满足：

1. **独立指令充足**：乱序执行引擎能找到足够多无依赖的指令来填充功能单元
2. **避免停顿**：减少 Load/Store 延迟（靠 Cache）、减少结构冒险（靠多功能单元）
3. **避免错误预测**：分支预测准确率高
4. **所有核心有工作**：程序提供足够的并发线程

---

### 2.5 Flynn 分类法

Michael Flynn 于 1966 年提出处理器体系结构的四类分类，至今仍是重要的概念框架：

| 类别       | 全称                                  | 含义            | 例子                 |
| -------- | ----------------------------------- | ------------- | ------------------ |
| **SISD** | Single Instruction, Single Data     | 经典串行处理器       | 单核 CPU             |
| **MIMD** | Multiple Instruction, Multiple Data | 多核处理器，每核独立执行  | 多核 CPU，GPU（在某种意义上） |
| **SIMD** | Single Instruction, Multiple Data   | 同一指令同时作用于多个数据 | 向量机、GPU            |
| **MISD** | Multiple Instruction, Single Data   | 实际中几乎不存在      | —                  |

**SIMD 是 ML 计算的核心**：矩阵乘法、卷积等运算本质上都是对大量数据施加相同操作，天然适合 SIMD 执行。GPU 可以看作极致化的 SIMD 机器。

---

## Module 3：Compute — SIMD 与 GPU

### 3.1 向量处理器（Vector Processor）

#### 向量寄存器

标量寄存器一次操作一个数据元素（64 bits）。向量寄存器存储一个**向量**——多个数据元素的集合：

|寄存器类型|大小|例子|
|---|---|---|
|标量（x86-64）|64 bits| `RAX` |
|AVX2 短向量|256 bits = 8×FP32| `YMM0` |
|AVX-512 短向量|512 bits = 16×FP32| `ZMM0` |
|Cray-1 长向量（1970s）|4096 bits = 64×FP64|—|

#### 向量指令的两种方向

**Vertical（垂直/逐元素）：** 两个向量逐元素计算，得到另一个向量。这是最常见的形式。

```
VA: [a b c d e]
VB: [t u v w x]
       ↓ VA + VB
VC: [a+t b+u c+v d+w e+x]
```

**Horizontal（水平）：** 对单个向量内部的元素进行操作，产生标量（如求最大值）或另一个向量（如 Permute）。

```
VA: [a b c d e]
       ↓ MAX(VA)
     MAX_SCALAR
```

#### 谓词/掩码（Predication / Masking）

向量机处理条件分支的方式：用一个 **Mask 向量**控制哪些 Lane 的结果有效：

```c
// 标量代码
for(i = 0; i < N; i++) {
    if(a[i] > 1)  c[i] = a[i];
    else          c[i] = 0;
}
```

对应的向量操作（以伪指令表示）：

```asm
vp  = va > v1      ; 生成 mask: [T F F T T]
@vp  vc = va       ; 仅 mask=T 的 Lane 写入
@!vp vc = v0       ; 仅 mask=F 的 Lane 写入零
```

```
va:   [2  1  0  4  5]
mask: [T  F  F  T  T]
vc:   [2  0  0  4  5]
```

没有条件跳转，所有 Lane 都执行（但 mask 控制哪些结果被提交）。这与 GPU 的 Warp Divergence 机制原理相同。

#### Gather / Scatter（间接访存）

- **Gather（间接 Load）**：向量寄存器中存储的是**地址**，从这些地址分散加载数据到向量寄存器中
- **Scatter（间接 Store）**：将向量中的数据写到向量寄存器所指定的多个离散地址

```
索引向量: [2  1  0  3  9]
内存:     [a  b  c  d  e  f  g  h  i  j]
                ↓ Gather
结果:     [c  b  a  d  j]
```

**性能影响**：Gather/Scatter 会访问不连续的内存地址，极易导致 Cache Miss，性能通常远差于连续 Load/Store。

#### x86 上的短向量 SIMD

实际 CPU 的向量宽度有限，常见规格：

|扩展集|向量宽度|支持 FP|
|---|---|---|
|SSE（2001）|128 bits|是（SSE 支持 IEEE 754 FP）|
|AVX/AVX2（2011/2013）|256 bits|是|
|AVX-512（2017）|512 bits|是|

**注意**：x86-64 的 SSE 之前的 MMX/3DNow! 等不完全支持 IEEE 754；只有 SSE 系列才支持符合标准的浮点运算。

#### 向量化的两种路径

**显式 SIMD intrinsics（手写）：**

```c
// 手写 AVX2 代码示例
__m256i avec0 = _mm256_load_si256(...);
__m256i bvec0 = _mm256_load_si256(...);
__m256i cvec  = _mm256_add_epi32(avec0, bvec0);
_mm256_store_si256(result, cvec);
```

代码繁琐、不可移植，但控制精确。

**编译器自动向量化（Autovectorization）：**

```c
// 普通 C 代码
void vec_add(int *A, int *B, int *C, int N) {
    for(int i = 0; i < N; i++)
        C[i] = A[i] + B[i];
}
```

使用 `gcc -O3` 编译后，编译器自动生成向量指令：

```asm
vmovdqu (%rdi,%rax), %ymm1
vpaddd  (%rsi,%rax), %ymm1, %ymm0
vmovdqu %ymm0, (%rdx,%rax)
```

前提：循环迭代之间没有数据依赖（编译器可以验证或程序员通过 `restrict` 声明保证）。

---

### 3.2 GPU 架构与 CUDA 编程模型

#### GPU 的历史起源

GPU 最初为 3D 游戏图形设计：图形渲染（光栅化、着色）计算量极大，但高度并行（每个像素独立计算）。约 2006 年，NVIDIA 将 GPU 改造为**通用并行处理器（GPGPU）**，统一着色器架构出现。

#### 现代 GPU 的架构特征

与 CPU 的对比：

| 维度   | CPU                        | GPU                           |
| ---- | -------------------------- | ----------------------------- |
| 核心数  | 几至几十个大核                    | 数千个小核（SM × CUDA Cores）        |
| 设计目标 | **低延迟**（latency-optimized） | **高吞吐**（throughput-optimized） |
| 执行方式 | 乱序超标量                      | In-order，但高度 SMT（64-way）      |
| 时钟频率 | 高（3–5 GHz）                 | 较低（1–2 GHz）                   |
| 缓存设计 | 大 L1/L2/L3 Cache           | 小 L1，大寄存器文件，Shared Memory     |
| 适用场景 | 复杂控制流，低延迟任务                | 大规模数据并行，吞吐量任务                 |

**在途线程数量**：以 NVIDIA A100 为例，每个 SM 最多 2048 个并发线程，共 108 个 SM，最多约 221,184 个线程同时在途。当一个 Warp 在等待内存时，调度器立即切换到另一个就绪 Warp，从而**隐藏内存延迟**。

#### CUDA 编程模型

CUDA 是一个**标量编程模型**：程序员写的像是串行的标量代码，但编译器和硬件将其在向量模式下执行（每个 "Thread" 实际是向量的一个 Lane）。

**核心概念层次：**

```
Grid（3D）
  └── Thread Block（3D，保证在同一 SM 内执行）
        └── Warp（32 个 Thread，实际执行单元，对应 SIMD 向量）
              └── Thread（程序员视角的单个执行单元）
```

**每个 Thread 通过 ID 确定自己处理哪块数据：**

```cuda
__global__ void vector_add(int *A, int *B, int *C, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N)
        C[tid] = A[tid] + B[tid];
}

// 调用：N 个元素，每个 Block 256 个线程
vector_add<<<(N + 255) / 256, 256>>>(A, B, C, N);
```

`threadIdx.x`：Block 内线程编号；`blockIdx.x`：Block 编号；`blockDim.x`：每个 Block 的线程数。

**关键约束**：CUDA 线程**不像 CPU 线程那样有独立的程序计数器**。硬件将同一 Warp 的 32 个线程当作一个向量指令执行。程序员看到的是 32 个独立线程的假象，但硬件实际是 SIMD。

#### Warp Divergence（Warp 散度）

当同一 Warp 的 32 个线程执行到条件分支，且部分线程走 true 分支、部分走 false 分支时，发生 Warp Divergence：

```cuda
// 假设一个 Warp 内 Thread 0-15 满足条件，16-31 不满足
if (tid % 2 == 0) {
    do_something();    // 只有 Thread 0,2,4... 执行，其他被 mask
} else {
    do_other_thing();  // 只有 Thread 1,3,5... 执行，其他被 mask
}
```

两个分支**串行**执行（先执行 true 分支，false 分支的 Lane 被 mask；再执行 false 分支），有效带宽减半。Warp 在汇合点（Join Point）重新合并。

> 编译器会在 Predication（直接 mask）和 Warp Divergence 机制之间选择；对于短分支倾向 Predication，对于长分支倾向 Warp Divergence。

#### Coalesced Memory Access（合并内存访问）

GPU 内存访问效率的核心：

**最优（Coalesced）：** Warp 内 32 个线程访问连续的内存地址 → 合并为一次宽内存事务，充分利用带宽

```cuda
// Good: thread i 访问 A[i]，32 个线程访问 A[0..31]（连续）
float val = A[threadIdx.x + blockIdx.x * blockDim.x];
```

**最差（Non-coalesced / Scatter）：** 每个线程访问不同的随机地址 → 32 次独立内存事务，带宽利用率 1/32

```cuda
// Bad: 每个线程访问随机地址（间接寻址）
float val = A[index[threadIdx.x + blockIdx.x * blockDim.x]];
```

实际缓存行（Cache Line）是最小传输单位，不一致地址会导致大量 Cache Miss。

#### GPU 使用的前提条件

以下情况下使用 GPU 才合算：

1. **有大量工作**：GPU 启动（kernel launch）有固定开销，小任务收益不抵开销
2. **工作是数据并行的**：所有数据可以用相同代码处理
3. **数据已在 GPU 上**（或 PCIe 传输开销可被计算收益覆盖）
4. **控制流简单**：复杂分支导致严重 Warp Divergence，效率急剧下降

---

### 3.3 向量编程实践

#### 核心原则

> **到达峰值算力的唯一途径是充分使用向量指令。**

一个不使用向量单元的程序，无论其他优化多好，都最多只能利用处理器理论算力的 $1/W$（$W$ 是向量宽度，对于 AVX2 + FP32 是 8）。

#### 实践建议

**1. GPU-first 设计：** 先让代码在 GPU 上高效运行，CPU 版本通常可以从中受益。

**2. 使用 DSL 而非裸 CUDA/intrinsics：** 高层域特定语言（如 OpenAI Triton、Google Pallas）比手写 CUDA 更易于获得高性能，且可移植：

```python
# Triton 示例：对向量加法的 tiled kernel
@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)
```

Triton 编译器会自动处理向量化、内存访问优化、寄存器分配等低层细节。

**3. 不要轻视 CPU 自动向量化：** 对于某些 ML 任务（如批量推理中的前处理），CPU 的 IREE 框架（Intermediate Representation Execution Environment）的自动向量化能提供相当竞争力的性能。

**4. ISPC（Intel Implicit SPMD Program Compiler）：** 在 CPU 上使用类似 CUDA 的 SPMD 编程模型，编译器负责将 "并行线程" 映射到 SIMD 指令，兼容 Intel、AMD、ARM。

---

## Module 4：数据表示

### 4.1 位运算与布尔代数基础

#### 物理层：位的表示

数字计算机中，位（Bit）在物理上通常是电压：

|逻辑值|TTL 电压|CMOS 电压|
|---|---|---|
|0（LOW）|0–0.8V|接近 0V|
|1（HIGH）|2–5V|接近 VCC（3.3V/5V）|

#### 基本逻辑门

|门|符号|真值表特征|
|---|---|---|
|**AND**| `&` |仅当两输入都为 1 时输出 1|
|**OR**| `\|` |至少一个输入为 1 时输出 1（包含或）|
|**XOR**| `^` |恰好一个输入为 1 时输出 1（排除或）|
|**NOT**| `~` |反转单个输入|
|**NAND**|—|NOT(AND)，**通用门**（可构造所有逻辑）|
|**NOR**|—|NOT(OR)，**通用门**|

NAND 和 NOR 是**通用门（Universal Gates）**：仅用 NAND（或仅用 NOR）可以实现任意布尔函数。

对 n 位向量的逻辑运算逐位独立进行：

```
0101 1001
AND
0111 0110
─────────
0101 0000
```

#### 数据单位

|大小（bits）|通用名称|
|---|---|
|8|byte（字节）|
|16|halfword / word（视架构而定）|
|32|word / doubleword（视架构而定）|
|64|word / quadword（视架构而定）|

**机器字长（Machine Word）** 是处理器一次可以操作的数据大小。64 位机器的字长为 64 bits。

#### 字节序（Endianness）

多字节数据在内存中的存储顺序：

```
存储 32 位值 0xDEADCAFE：

地址:       x      x+1    x+2    x+3
Big-Endian: 0xDE   0xAD   0xCA   0xFE   (高位在低地址)
Little-Endian: 0xFE   0xCA   0xAD   0xDE   (低位在低地址)
```

- Intel/AMD x86 通常是 **Little-Endian**
- 网络协议使用 **Big-Endian**（"网络字节序"）
- ARM 可切换

#### 位的解释

**同一位模式可以有不同含义**：字节 `0x55` 既可以是整数 85，也可以是 x86 汇编指令 `push %rbp`。位本身没有类型，解释（Interpretation）由上下文决定。这是一个强大的基础概念（代码即数据）。

---

### 4.2 整数表示

#### 有符号整数的三种编码方案

给定 n 位，需要表示正数和负数。历史上有三种方案：

**方案一：符号-数值（Sign-Magnitude）**

最高位（MSB）为符号位（0 正 1 负），其余位为数值：

```
+89₁₀ = 0101 1001
-89₁₀ = 1101 1001   (仅符号位变化)
+0   = 0000 0000
-0   = 1000 0000   ← 有两个零！
```

8 位范围：-127 到 +127（有 +0 和 -0 两种表示）。加减法电路复杂。

**方案二：一补数（One's Complement）**

对正数取反（每位翻转）得到对应负数：

```
+89₁₀ = 0101 1001
-89₁₀ = 1010 0110   (按位取反)
+0   = 0000 0000
-0   = 1111 1111   ← 仍然有两个零
```

范围：-127 到 +127。加法需要处理末位进位回卷（end-around carry）。

**方案三：二补数（Two's Complement）**⭐ 现代计算机通用方案

对正数取反后加 1（即取一补数再加 1）：

```
+89₁₀ = 0101 1001
-89₁₀ = 1010 0111   (取反后加1: 1010 0110 + 1 = 1010 0111)
 0   = 0000 0000   ← 唯一的零！
-128₁₀= 1000 0000   (无对应正数，范围不对称)
```

8 位范围：**-128 到 +127**（非对称，但只有一个 0）。加减法电路与无符号整数相同，无需特殊处理，这是它被普遍采用的原因。

C23 标准明确要求使用二补数；之前的标准允许其他实现，但实际上所有主流处理器都用二补数。

#### C 语言整数类型

```c
int8_t  a;    // 有符号 8 位，范围 -128 ~ 127
uint8_t ua;   // 无符号 8 位，范围 0 ~ 255
int32_t b;    // 有符号 32 位，范围 -2³¹ ~ 2³¹-1
uint64_t uc;  // 无符号 64 位，范围 0 ~ 2⁶⁴-1
```

#### 位域（Bitfields）

当空间宝贵时，可以将多个小字段打包进一个机器字中：

```c
// 将"日期"(1-31, 需5位)和"星期"(0-6, 需3位)打包进一个 uint8_t
uint8_t daydow;
daydow = (day << 3) | dow;   // day 占高 5 位，dow 占低 3 位

// 提取星期
dow = daydow & 0x7;          // 用 mask 保留低 3 位

// 提取日期
day = (daydow >> 3) & 0x1f;  // 右移 3 位，再 mask 高位
```

位域在 ML 中的应用：量化权重的存储（INT4 将两个权重打包进一个 byte）、GPU 硬件内部的各种状态标志（如 Unix 文件权限位）。

---

### 4.3 IEEE 754 浮点标准

#### 浮点数的动机

实数有无穷多个，无法用有限位精确表示。解决方案是使用**二进制科学计数法**：

$$\text{value} = \text{significand} \times 2^{\text{exponent}}$$

例如：$1.011_2 \times 2^3 = 1110_2 = 14_{10}$

#### IEEE 754 单精度（FP32）格式

32 位，分三个字段，按 **s | E | M** 顺序排列：

```
位 31   30-23     22-0
   s  |  E(8)  |  M(23)
```

|字段|位数|含义|
|---|---|---|
|**s**（符号位）|1|0 正 1 负|
|**E**（偏置指数）|8|实际指数 + 127（偏置值 Bias）|
|**M**（尾数）|23|小数点后的 23 位，隐含前导 1|

**隐含前导 1（Implicit Leading 1）：** 归一化浮点数的尾数总是形如 $1.xxx...$，这个前导 1 不存储，因此有效精度是 24 位（约 7 位十进制有效数字）。

**偏置指数（Biased Exponent）：** 实际指数 $e$ 存储为 $e + 127$，范围是 1–254（代表 $-126$ 到 $+127$）。使用偏置表示的好处是可以用整数比较操作直接比较浮点数的大小（假设符号位相同）。

**浮点数范围：**

- 最小正数（归一化）：$1.0 \times 2^{-126} \approx 1.18 \times 10^{-38}$
- 最大正数：$\approx 3.4 \times 10^{38}$

#### 特殊值

|情形|s|E|M|值|
|---|---|---|---|---|
|正零|0|全 0|全 0|+0|
|负零|1|全 0|全 0|-0|
|+∞|0|全 1|全 0|+∞|
|-∞|1|全 1|全 0|-∞|
|NaN|×|全 1|≠0|NaN（任何涉及 NaN 的运算仍得 NaN）|
|非规格化数|×|全 0|≠0|Denormal（允许"渐进下溢"）|

**NaN 的产生**：`0/0`、`∞/∞`、`√(-1)` 等未定义操作产生 NaN。NaN 具有**传播性**：任何以 NaN 为操作数的运算结果仍是 NaN。

**非规格化数（Denormals）的意义：**

```
a = 1.000...000 × 2^{-126}
b = 1.000...001 × 2^{-126}

a - b = 0.111...111 × 2^{-126}  ← 这是一个非规格化数
```

没有 Denormal 的系统会将这个结果直接置零（下溢到零），导致 $a = b$ 而实际上 $a \neq b$，进而可能引发除零错误。IEEE 754 的渐进下溢机制通过允许非规格化数来避免这个问题。

> **性能注意**：在某些处理器上（特别是 GPU），处理 Denormal 的运算在软件中完成，速度极慢（相差数十倍）。ML 训练中应确保数值不进入非规格化区间。

#### 浮点数的陷阱

**精度损失示例：**

```c
float f = 16777216.0;   // 即 2^24，恰好是 FP32 尾数精度的上限
f = f + 1.0;
printf("%f\n", f);      // 输出：16777216.000000，不是 16777217！
```

原因：$16777216 + 1 = 16777217$ 在 FP32 中无法精确表示（需要超过 24 位尾数），被舍入回 $16777216$。

**非结合性：** $(a + b) + c \neq a + (b + c)$（通常情况）。浮点加法的舍入误差取决于运算顺序。这对 ML 训练的数值稳定性有重要影响（尤其是梯度累积）。

**四种舍入模式：**

|模式|说明|默认？|
|---|---|---|
|Round to Nearest (Even)|四舍五入，平局取偶数|✓ 默认|
|Round toward Zero|截断（向零）||
|Round toward +∞|向上取整||
|Round toward -∞|向下取整||

#### FP32 vs FP16 vs BF16 vs FP8

ML 对精度要求比科学计算低，因此可使用更少位数的浮点格式：

|格式|总位数|指数位|尾数位|有效十进制数字|用途|
|---|---|---|---|---|---|
|**FP64**（double）|64|11|52|~17|科学计算|
|**FP32**（float）|32|8|23|~7|训练（传统），推理|
|**FP16**|16|5|10|~3|混合精度训练（前向/反向），推理|
|**BF16**|16|8|7|~2–3|Google TPU，减少溢出风险（指数位与 FP32 相同）|
|**FP8**|8|—|—|~1–2|最新推理优化（NVIDIA H100/B200）|

**BF16 的优势**：保留 FP32 的 8 位指数（相同数值范围，不容易上溢/下溢），仅裁减尾数。适合梯度计算，无需额外调整学习率等超参数。

**混合精度训练（AMP）** 的典型配置：

- 前向传播和反向传播使用 FP16
- 梯度和参数更新使用 FP32 主权重（Master Weights）
- 这样可以减少 2× 内存占用，同时利用 Tensor Core 的 FP16 算力

---

### 4.4 复合数据结构（Structs & Arrays）

#### 结构体内存布局

```c
struct node {
    int value;          // 4 bytes
    struct node *left;  // 8 bytes (64-bit pointer)
    struct node *right; // 8 bytes
};
```

**紧密打包（Tight Packing）：** 总共 20 bytes

```
[value:4][left:8][right:8] = 20 bytes
```

**带对齐填充（Padding）：** 实际常见布局

```
[value:4][pad:4][left:8][right:8] = 24 bytes
```

编译器遵循**自然对齐规则**：每个字段的地址必须是其大小的整数倍（例如 8 字节指针必须在 8 字节边界对齐）。结构体整体大小是最大字段大小的整数倍。

**为什么对齐重要？** 不对齐的内存访问在某些架构上会引发异常（硬件要求），在其他架构上会降低性能（需要多次内存访问）。

#### 多维数组的内存布局

内存是一维的。二维数组 `A[M][N]` 映射到线性内存有两种方案：

**Row-major（行优先，C/C++/Python NumPy 默认）：**

```
A[0][0] A[0][1] A[0][2] | A[1][0] A[1][1] A[1][2] | ...
```

访问公式：`index = row × COLS + col`

**Column-major（列优先，Fortran/MATLAB/Julia 默认）：**

```
A[0][0] A[1][0] A[2][0] | A[0][1] A[1][1] A[2][1] | ...
```

访问公式：`index = col × ROWS + row`

**性能影响（举例）：**

```c
// 矩阵转置：B[j][i] = A[i][j]
for(i = 0; i < M; i++)
    for(j = 0; j < N; j++)
        B[j][i] = A[i][j];   // A 按行读（连续），B 按列写（不连续！）
```

如果 A 存储为 row-major（C 默认），读 A 时连续（Cache 友好），写 B 时跳步（Cache 不友好）。这是 Cache 性能优化（Loop Tiling）的经典例子。

#### 联合体（Union）

```c
union intvar {
    char  c;    // 1 byte
    short s;    // 2 bytes
    int   i;    // 4 bytes
    long  l;    // 8 bytes
};
```

Union 的所有字段在内存中**重叠**，总大小等于最大字段的大小（此例为 8 bytes）。写入一个字段后读取另一个字段的行为是实现定义（Implementation-Defined）的，但常被用于类型双关（Type Punning）技巧，如查看 float 的原始位表示：

```c
union float_bits {
    float f;
    uint32_t i;
};
union float_bits fb;
fb.f = 3.14f;
printf("0x%08X\n", fb.i);   // 输出浮点数的原始位模式
```

---

### 4.5 稀疏数据结构

#### 为什么稀疏格式重要

**稀疏矩阵**中非零元素（NNZ）数量远少于总元素数量。用稠密矩阵存储稀疏矩阵既浪费存储，又浪费计算（大量乘以零的操作）。

ML 中的稀疏场景：

- 图神经网络的邻接矩阵（通常极稀疏）
- Mixture of Experts 中的 Gating 矩阵（稀疏激活）
- 模型剪枝（Pruning）后的权重矩阵
- 词嵌入 Lookup（One-hot 输入本质上是稀疏的）

#### COO（Coordinate Format）格式

用两个（或三个）数组存储非零元素的坐标（和值）：

```
稀疏矩阵:             COO 表示:
0  0  1  0            row: [0  2  2]
0  0  0  0    →      col: [2  1  3]
0  1  0  1            (可选 data 数组)
0  0  0  0
```

**优点**：直观，易于构建。
**缺点**：没有对某一行的快速访问，存储冗余（row 数组存储大量重复值）。

#### CSR（Compressed Sparse Row）格式

三个数组：`col`（列索引）、`row_start`（每行起始位置）、`data`（可选，非零值）：

```
稀疏矩阵:             CSR 表示:
0  0  1  0            col:       [2  1  3]
0  0  0  0    →      row_start: [0  1  1  3  3]
0  1  0  1            (第 i 行的非零元素在 col[row_start[i]:row_start[i+1]])
0  0  0  0
```

**访问第 i 行的所有非零元素：**

```c
for(j = row_start[i]; j < row_start[i+1]; j++) {
    column = col[j];
    value  = data[j];   // 处理 A[i][column] = value
}
```

**CSR 的性能问题：**

- **间接内存访问**：先加载 `row_start[i]`，再加载 `col[j]`，再加载 `data[j]` ——多次 Cache Miss
- **难以向量化**：每次迭代的内存地址不连续（Gather 访问）
- **不支持高效修改**：向零元素写入非零值需要重建整个结构

> 为此衍生出了 ELLPACK（ELLR）等块稀疏格式，以更规整的访问模式换取更好的向量化支持。

---

### 4.6 数据布局的性能含义

#### AoS vs SoA

这是 ML 系统中最重要的数据布局决策之一：

**AoS（Array of Structures，结构体数组）：**

```c
struct Point { float x; float y; };
struct Point pts[1000];
// 内存: [x0 y0][x1 y1][x2 y2]...
```

**SoA（Structure of Arrays，数组结构体）：**

```c
struct Points { float *x; float *y; };
struct Points p;
p.x = malloc(1000 * sizeof(float));
p.y = malloc(1000 * sizeof(float));
// 内存: [x0 x1 x2 ... x999][y0 y1 y2 ... y999]
```

**性能对比（处理所有点的 x 坐标）：**

| 布局  | 访问模式                    | 向量化          | Cache 效率    |
| --- | ----------------------- | ------------ | ----------- |
| AoS | 每隔 sizeof(Point) 访问一个 x | 需要 Gather，困难 | 差（加载不需要的 y） |
| SoA | 连续访问 x 数组               | 直接，高效        | 好           |

ML 框架（如 PyTorch 的 Tensor）通常采用 SoA 思维——不同通道（channels）存储在连续内存中，以优化向量化和 Cache 利用。

#### Row-major vs Column-major 对矩阵乘法的影响

矩阵乘法 $C = A \times B$ 的朴素实现：

```c
for(i = 0; i < M; i++)
    for(j = 0; j < N; j++)
        for(k = 0; k < K; k++)
            C[i][j] += A[i][k] * B[k][j];
```

对于 Row-major 存储：

- 访问 `A[i][k]`：沿 k 变化，连续（Cache 友好）
- 访问 `B[k][j]`：沿 k 变化，按列跳跃（Cache 不友好！）

通过 **Loop Interchange** 改变循环顺序（i, k, j），或 **Loop Tiling / Blocking** 将矩阵分成小块后处理，可以大幅提升 Cache 命中率——这是 Triton 等高性能 ML 编译器的核心技术。

---

## Module 5：内存系统

### 5.1 内存技术与度量指标

#### 内存技术对比

| 技术           | 易失性 | 速度（延迟）            | 密度                | 用途                     |
| ------------ | --- | ----------------- | ----------------- | ---------------------- |
| **寄存器**      | 是   | 最快（< 1 ns）        | 最低                | CPU/GPU 内部临时存储         |
| **SRAM**     | 是   | 极快（~ 1–5 ns）      | 低（6 晶体管/bit）      | CPU/GPU 的 Cache（L1/L2） |
| **DRAM**     | 是   | 慢（~ 50–100 ns）    | 高（1 晶体管 + 电容/bit） | 主内存（RAM），GPU VRAM      |
| **HBM**      | 是   | 快（DRAM 级延迟，但带宽更高） | 高                 | 现代 GPU 的主内存（A100/H100） |
| **NVMe SSD** | 否   | 很慢（~ 100 μs）      | 很高                | 持久化存储，模型权重加载           |
| **HDD**      | 否   | 极慢（~ 10 ms）       | 最高                | 大规模数据集存储               |

#### SRAM（Static RAM）

6 个晶体管构成一个双稳态锁存器存储 1 比特，不需要刷新，读取不破坏数据：

```
VDD
 |
M5─M6
|   |
M2  M4
|   |
M1  M3
|   |
WL───── (Word Line, 控制行选择)
BL BL̄  (Bit Line, 读/写数据)
```

SRAM 单元面积大（因此成本高），但速度极快，是 Cache 的理想选择。较新的 Cache 设计可能使用嵌入式 DRAM（eDRAM），密度更高但速度略慢。

#### DRAM（Dynamic RAM）

1 晶体管 + 1 电容存储 1 比特。电容上的电荷随时间泄漏，需要**周期性刷新（Refresh）**。每次读取后电荷丢失，需要**读后恢复（Restore）**。

VRAM（显存）在技术上与 DRAM 相同，差异主要在接口和时序优化上。

**DRAM 的访问延迟层次（以 DDR4 为例）：**

- Row 激活（tRCD）：~13–15 ns
- CAS 延迟（tCL）：~13–15 ns
- 完整随机访问：~50–80 ns

#### HBM（High Bandwidth Memory）

现代 GPU（NVIDIA A100、H100、AMD MI300X）的标准内存技术：

```
┌──────────────────────────┐
│     GPU Die（计算）       │
├──────────────────────────┤ ← 超宽接口 (1024–2048 bits)
│   HBM Stack（多层 DRAM）  │ ← 硅通孔（TSV）连接各层
│   Layer 0                │
│   Layer 1                │
│   Layer 2                │
│   Layer 3                │
└──────────────────────────┘
整体封装在同一封装基板（Silicon Interposer）上
```

|规格|HBM2|HBM2e|HBM3|
|---|---|---|---|
|接口宽度|1024 bits/stack|1024 bits|2048 bits|
|带宽/stack|~256 GB/s|~307 GB/s|~819 GB/s|
|NVIDIA GPU 采用|V100|A100|H100|

与 GDDR6（PCIe GPU 使用的传统显存）相比，HBM 通过更宽的接口提供更高带宽，而不是更高的时钟频率，从而避免信号完整性问题。

#### 三个核心度量指标

**密度（Density）：** 每单位面积/体积的比特数。越高越好（意味着更多存储，成本更低）。

**延迟（Latency）：** 从发出请求到收到数据的时间（纳秒或时钟周期）。越低越好。注意延迟与带宽的区别：一次大块传输的延迟可能与一次小块传输相同，但吞吐不同。

**带宽（Bandwidth）：** 单位时间内可传输的数据量（GB/s）。越高越好。

---

### 5.2 内存组织的通用模型

#### 串行内存 Bank 模型

每个 Memory Bank（内存分区）一次只能处理**一个**请求，通过队列缓冲多个请求：

```
请求队列 → [Bank 0] → 响应
          [Bank 1]
          [Bank 2]
          ...
控制器
```

**Memory Level Parallelism（MLP）：** 同时在途的内存请求数量。要充分利用内存带宽，需要有足够多的并发请求来"填满"内存子系统的延迟-带宽乘积：

$$\text{并发请求数} = \text{带宽} \times \text{延迟} / \text{请求大小}$$

例如：带宽 50 GB/s，延迟 100 ns，64 字节请求 → 需约 78 个并发请求才能充分利用带宽。

#### Partition Camping（分区热点问题）

如果程序的内存访问模式总是集中在同一个 Bank，其他 Bank 空闲，整体带宽利用率极低。内存控制器通过**地址交错（Interleaving）** 将连续地址分散到不同 Bank，缓解这个问题：

```
地址 0x0000 → Bank 0
地址 0x0040 → Bank 1
地址 0x0080 → Bank 2
...
```

#### GPU Shared Memory 的 Bank Conflicts

GPU 的 Shared Memory（片上 Scratchpad）也有 32 个 Bank（对应 Warp 的 32 个 Lane）。当同一 Warp 的多个线程访问同一个 Bank 的不同地址时，发生 **Bank Conflict**，这些访问被串行化，性能急剧下降。

```
Bank 0   Bank 1   Bank 2  ...  Bank 31
地址 0    地址 1    地址 2  ...  地址 31
地址 32   地址 33   地址 34 ...  地址 63
...
```

若 Thread 0 访问地址 0，Thread 1 访问地址 32（同一 Bank 0），则产生 2-way Bank Conflict。

#### 内存访问模式的三个例子

```c
// 例1：高 MLP，Cache 友好
for(row=0; row < NROWS; row++)
    for(col=0; col < NCOLS; col++)
        out[row * NCOLS + col] = in[col * NROWS + row];
// 读 in 是按列跳步（stride access），可预取但 Cache miss 高

// 例2：间接访问（稀疏图遍历）
for(row = 0; row < NROWS; row++) {
    for(j = row_start[row]; j < row_start[row+1]; j++) {
        sum += edgedata[col[j]];   // col[j] 不可预测，高 Cache miss
    }
}

// 例3：指针追逐（Pointer Chasing，最差情形）
current = head;
while(current) {
    if(current->value == search) return current;
    current = current->next;   // 每次访问地址取决于上次读取结果
}
// 完全无法预取，MLP = 1，每次访问都要等待完整 DRAM 延迟
```

---

### 5.3 Cache 层次结构

#### 为什么需要 Cache

内存延迟（~100 ns）与 CPU 速度（~1 ns/cycle @ 1 GHz）之间存在巨大差距（"Memory Wall"）。Cache 是速度与容量之间的折中：

```
距 CPU 的距离（速度快 ← → 慢）：
寄存器（<1ns, ~KB） → L1 Cache（~4ns, ~32KB） → L2 Cache（~12ns, ~256KB）
→ L3 Cache（~40ns, ~16MB） → DRAM（~100ns, ~GB） → NVMe（~100μs, ~TB）
```

#### Cache 的工作原理

当 CPU 加载一个地址的数据时：

1. 查 L1 Cache：**Hit** → 1–4 个周期返回数据
2. L1 Miss → 查 L2：**Hit** → ~12 周期返回数据，同时填入 L1
3. L2 Miss → 查 L3：**Hit** → ~40 周期返回数据
4. L3 Miss → 查 DRAM：**Miss** → ~100–200 周期返回数据

**平均内存访问时间（AMAT）：**

$$T_{avg} = H_{L1} \cdot T_{L1} + (1 - H_{L1})\left[H_{L2} \cdot T_{L2} + (1 - H_{L2}) \cdot T_{DRAM}\right]$$

**示例**：假设 $H_{L1} = 95\%$，$T_{L1} = 4$ 周期，$H_{L2} = 80\%$，$T_{L2} = 12$ 周期，$T_{DRAM} = 100$ 周期：

$$T_{avg} = 0.95 \times 4 + 0.05 \times (0.80 \times 12 + 0.20 \times 100) = 3.8 + 0.05 \times (9.6 + 20) = 3.8 + 1.48 \approx 5.3 \text{ 周期}$$

若 Cache 完全有效（$H_{L1} = 100\%$），则 $T_{avg} = 4$ 周期；若全部 Miss，则 $T_{avg} = 100$ 周期。Cache 有效时性能提升 **25×**。

#### 局部性原理（Locality of Reference）

Cache 有效的根本原因是大多数程序表现出**局部性**：

**空间局部性（Spatial Locality）：** 如果地址 $x$ 被访问，则附近地址（$x+1, x+2, \ldots$）很快也会被访问。

- 利用方式：以 **Cache Line**（通常 64 字节）为单位传输数据，一次从 DRAM 读取连续的 64 字节到 Cache

**时间局部性（Temporal Locality）：** 如果地址 $x$ 被访问，则近期会再次访问 $x$。

- 利用方式：将 $x$ 保留在 Cache 中等待再次访问（Cache 替换策略管理这个过程）

**代码中的局部性示例：**

```c
// 良好的空间局部性：顺序访问数组
for(i = 0; i < N; i++) {
    if(a[i] > max) max = a[i];   // a[0], a[1], a[2]... 连续
}

// 破坏空间局部性：随机间接访问
for(i = 0; i < N; i++) {
    if(a[b[i]] > max) max = a[i];  // a[b[i]]，b[i] 随机，Cache miss 率高 10× 
}
```

实测（同一机器）：顺序访问约 13ms；随机间接访问约 65ms，相差 **5×**。

#### Cache 组织方式

**Direct-Mapped（直接映射）：** 每个内存地址只能映射到 Cache 中的固定一个位置（地址的某几位决定）。实现最简单，但同一 Cache Set 内不同地址之间会**冲突**（Conflict Miss）。

**Fully Associative（全相联）：** 任何内存地址可以存储在 Cache 的任意位置。冲突 Miss 最少，但查找硬件复杂（需要并行比较所有 Cache 条目）。

**Set-Associative（组相联）：** 折中方案。将 Cache 分为若干 Set，每个 Set 内部全相联（K-way）。N-way Set-Associative Cache 在硬件复杂度和 Conflict Miss 率之间平衡，是现代 CPU Cache 的标准设计。

```
Set-Associative Cache（4-way，4个Set）：
Set 0: [行0] [行1] [行2] [行3]  ← 4 个行可存任何映射到Set 0的地址
Set 1: [行4] [行5] [行6] [行7]
Set 2: [行8] ...
Set 3: ...
```

#### 3C Cache Miss 模型

|类型|名称|原因|消除方式|
|---|---|---|---|
|**Compulsory Miss**|强制 Miss|数据第一次访问，Cache 中必然没有|预取（Prefetching）|
|**Conflict Miss**|冲突 Miss|在 Direct-Mapped 或 Set-Associative 中，多个地址竞争同一 Set|增加相联度|
|**Capacity Miss**|容量 Miss|Cache 容量不足以存放工作集|增大 Cache 或减小工作集（Tiling）|

#### Cache 替换策略

当 Cache 满需要驱逐一个条目时：

- **OPT（Optimal / Bélády's）：** 驱逐未来最晚被再次使用的行。理论最优，但需要未来知识，只用于评估基准。
- **LRU（Least Recently Used）：** 驱逐最近最少使用的行。近似 OPT 的最常用策略，实现有一定硬件开销。
- **MRU（Most Recently Used）：** 驱逐最近刚使用的行。听起来反直觉，但对于扫描型访问模式（Scan-Once）反而有效（避免污染 Cache）。
- **其他**：FIFO、Random、LIRS 等，各有适用场景。

#### Cache 写策略

**Write-Through（写直通）：** 写操作同时更新 Cache 和下一级内存。简单但每次写都产生内存流量。

**Write-Back（写回）：** 写操作只更新 Cache，标记为"脏（Dirty）"，仅在该 Cache 行被驱逐时才写回下一级内存。减少内存流量，但实现更复杂（需要 Dirty Bit）。

现代 CPU 通常使用 Write-Back；某些 GPU 情景（如写入后不再读取的输出数据）可使用写直通模式。

#### GPU 内存层次的特殊性

GPU 的内存层次与 CPU 有一个重要区别——**寄存器堆大于 L1 Cache**：

```
CPU 内存层次（大→小）：DRAM >> LLC >> L2 >> L1 >> 寄存器
GPU 内存层次（大→小）：HBM >> L2 >> 寄存器/Shared Mem >> L1

GPU A100 典型值：
  HBM:         80 GB（带宽 ~2 TB/s）
  L2 Cache:    40 MB
  Shared Mem:  每个 SM 共享，128 KB
  寄存器:      每个 SM 256 KB（远大于 L1 Cache！）
```

**Scratchpad Memory / Shared Memory（片上暂存器）：** GPU 的 Shared Memory 是程序员显式管理的片上 SRAM，需要手动加载（`__shared__` 修饰符 + `tl.load`），不透明于程序员（不像 CPU Cache 自动工作）。但速度极快（与寄存器相当），是 CUDA/Triton 高性能 kernel 的关键资源。

**DMA（Direct Memory Access）：** CPU 的 DMA 控制器（以及 GPU 的专用 Copy Engine）允许异步内存传输：程序指定源地址、目标地址和长度，DMA 引擎独立完成数据移动，完成后通知 CPU（或 GPU）。这样 CPU/GPU 可以在数据传输期间继续处理其他工作，实现**计算与通信重叠（Overlap）**。

---

### 5.4 Cache Coherence 与虚拟内存

#### Cache Coherence（Cache 一致性）

多核处理器中，每个核心有自己的 L1/L2 Cache，但共享 L3 Cache 和主内存。问题：

```
Core 0 的 L1 Cache: [x = 5]   ← Core 0 修改了 x，但只写到 L1 Cache
Core 1 的 L1 Cache: [x = 3]   ← Core 1 读到的是旧值
主内存:        [x = 3]    ← 主内存也是旧值
```

**Cache 一致性**要求：所有处理器对同一内存地址看到一致的值。

**MESI 协议**（Modified, Exclusive, Shared, Invalid）是最广泛使用的 Cache Coherence 协议。每个 Cache 行有四种状态：

|状态|含义|
|---|---|
|**Modified（M）**|该行已被本核修改，且与主内存不一致；其他核的 Cache 中无此行有效副本|
|**Exclusive（E）**|本核有此行的唯一有效副本，与主内存一致，但其他核的 Cache 中无此行|
|**Shared（S）**|多个核的 Cache 中有此行的副本，且与主内存一致（只读状态）|
|**Invalid（I）**|此行的内容无效，下次访问必须从其他级别重新加载|

**对 ML 的影响**：在多核/多 GPU 训练中，频繁共享和更新的数据（如梯度 AllReduce 中的参数）会引发大量 Cache Coherence 流量。这也是为什么 ML 框架倾向于避免细粒度共享，而是使用大块 AllReduce。

#### 虚拟内存（Virtual Memory）

**核心问题**：多个程序同时运行时，如何让每个程序都认为自己独占整个内存地址空间？

**解决方案**：每个进程使用**虚拟地址（Virtual Address）**，操作系统和硬件协作将其翻译为**物理地址（Physical Address）**。

```
进程 A 虚拟地址空间（假设 64-bit）:
0x0000000000000000 ─── 0xFFFFFFFFFFFFFFFF
        │
        │ 地址翻译（Page Table Walk / TLB Lookup）
        ↓
物理内存（实际 RAM，如 16 GB）:
0x0000000000000000 ─── 0x000000003FFFFFFF
```

**Page（页面）** 是虚拟内存管理的基本单位，通常为 4 KB（也有 2 MB 大页）。

**Page Table（页表）** 存储虚拟页号 → 物理帧号的映射。每个进程有独立的页表，由操作系统维护。

**TLB（Translation Lookaside Buffer）** 是页表的 Hardware Cache，存储最近的地址翻译结果：

```
虚拟地址 → [TLB 查找]
             ├── TLB Hit：直接得到物理地址（1–3 个周期）
             └── TLB Miss → [Page Table Walk（多次内存访问）]
                              ├── 页存在 → 得到物理地址（填入 TLB，约 20–50 周期）
                              └── Page Not Present → Page Fault（触发 OS 处理）
```

**Page Fault（缺页异常）** 有两种情形：

1. 页在内存中但 Page Table 条目不存在：OS 更新 Page Table，返回
2. 页不在内存中（已被换出到 Swap）：OS 从 Disk 读取页面，Page Table，约 10 ms 延迟

**内存保护（Memory Protection）：** 每个 Page Table 条目包含权限位（Read/Write/Execute），OS 以此实现进程间隔离：进程 A 无法访问进程 B 的内存（除非通过共享内存机制）。这是操作系统安全模型的基础。

**对 ML 系统的影响：**

CUDA 程序在 GPU 上运行时，GPU 内存也使用虚拟地址（Unified Virtual Addressing, UVA）。大型模型（几十 GB）会占用大量 TLB 条目；使用 **Huge Pages（大页，2 MB 或 1 GB）** 可以减少 TLB Miss，对 ML 推理延迟有可见改善（某些情况下 5–15%）。

---

## Part I 总结

### 核心概念速查

|概念|核心公式/原理|ML 系统关联|
|---|---|---|
|**执行时间**| $T = W \times t / P$ |衡量 kernel 的并行效率|
|**流水线 IPC**|停顿由冒险（HAZ）引入|GPU in-order 流水，需足够并发隐藏延迟|
|**SIMD 向量化**|单指令处理 W 个数据|Triton、AVX-512 等 ML kernel 的基础|
|**AMAT**| $H \cdot T_{hit} + (1-H) \cdot T_{miss}$ |分析 kernel 是 compute-bound 还是 memory-bound|
|**Cache 局部性**|Spatial + Temporal Locality|Tiling（分块）优化的核心目标|
|**IEE 754**|s\|E\|M，偏置指数，隐含前导 1|量化（FP16/BF16/FP8）的设计基础|
|**二补数**|取反加一，唯一零，非对称范围|INT8/INT4 量化的数值处理|
|**CSR / Sparse**| `col[]` + `row_start[]` 间接访问|稀疏注意力、MoE Gating 的存储格式|
|**虚拟内存/TLB**|VA→PA 翻译，TLB 缓存|大模型的 Huge Page 优化|
|**DMA**|异步内存传输|GPU H2D/D2H 传输，计算-通信 Overlap|

### Part I → Part II 的衔接

Part I 建立的硬件模型将直接用于 Part II 的性能分析：

- CPU 流水线的 IPC 分析 → Roofline 模型的计算上界 $T_{peak}$
- Cache 层次结构和 AMAT → Roofline 模型的带宽上界 $\beta$
- SIMD 宽度和 GPU Warp 尺寸 → 理解算子库（cuBLAS, Triton）的性能特征
- 内存带宽与延迟 → 判断 ML 算子是 compute-bound 还是 memory-bound
- 虚拟内存与 DMA → 理解 PyTorch 的 Pin Memory、CUDA Stream 等优化

# Part II：ML 程序的系统性执行（Real-World Efficiency）

## Module 7：ML 程序作为 Loop 密集代码

### 7.1 Loop 优化的动机

#### 程序热点与 Loop

任何程序的执行时间都高度集中在少数"热点（Hot Spots）"代码——通常是循环体。80/20 法则：20% 的代码消耗 80% 的运行时间。

ML 算子为何以 Loop 为核心：ML 程序大量使用数组、矩阵、张量（Tensor），几乎所有算子（Conv、MatMul、Attention）都可以展开为多层嵌套循环：

```c
// 矩阵乘法 C = A × B 的朴素实现
for(int i = 0; i < M; i++)          // 输出行
    for(int j = 0; j < N; j++)      // 输出列
        for(int k = 0; k < K; k++)  // 累加维度
            C[i][j] += A[i][k] * B[k][j];
```

**Loop 优化的两个方向：**

|方向|目标|关联 Part I 概念|
|---|---|---|
|**Front-end 优化**|产生足够多的独立工作，填满 CPU/GPU 流水线和核心|ILP、MLP、TLP（指令/内存/线程级并行）|
|**Back-end 优化**|减少执行每项工作的实际开销|SIMD 向量化、Cache 局部性（Spatial/Temporal）|

### 7.2 单循环变换

#### Loop Unrolling（循环展开）

将循环体复制若干次，减少循环本身的开销（branch、循环计数器更新），并为编译器提供更大的调度窗口：

```c
// 原始循环
for(i = 0; i < N; i++) {
    c[i] = a[i] * b[2*i];
}

// 展开因子 U = 4
for(i = 0; i < N/4; i += 4) {
    c[i]   = a[i]   * b[2*i];
    c[i+1] = a[i+1] * b[2*(i+1)];
    c[i+2] = a[i+2] * b[2*(i+2)];
    c[i+3] = a[i+3] * b[2*(i+3)];
}
// 尾处理：处理 N 不整除 4 的剩余元素
```

**收益**：

- 减少每次迭代的 Branch 指令（`N` 次减为 `N/4` 次）
- 使独立运算（`c[i]` 和 `c[i+1]` 彼此独立）对乱序执行器可见
- 为向量化铺路（四条独立的乘法 → 一条 SIMD 乘法指令）

**代价**：增大代码体积（指令 Cache 压力）；展开因子过大会用尽寄存器（Register Pressure）。

#### Loop Splitting / Peeling（循环分裂/剥离）

将一个循环拆分为多个部分，通常用于处理边界条件（使主循环的输入满足对齐或整除要求）：

```c
// 展开后可能有尾部不整除的问题
for(i = 0; i < N/4; i += 4) {
    // 主循环，N/4 对齐部分
}
for(; i < N; i++) {
    // 尾部剩余元素，逐个处理
}
```

实际 ML 编译器（Triton、TVM）大量使用 Splitting 来处理矩阵维度不整除 Tile 大小的情形（Mask 机制）。

#### Loop Vectorization（循环向量化）

将循环迭代合并为 SIMD 指令：

```c
// 标量版本（概念上每次处理 1 元素）
for(i = 0; i < N; i++)
    c[i] = a[i] * b[2*i];

// 向量化版本（以伪指令表示，每次处理 4 元素）
for(i = 0; i < N/4; i += 4) {
    simd_value_4 av = simd_load_4(a + i);     // 加载 4 个 a 元素
    simd_value_4 bv = simd_gather_4(b, 2*i);  // gather，因为步长为 2
    simd_value_4 cv = simd_mul_4(av, bv);
    simd_store_4(c + i, cv);
}
```

**向量化的前提**：循环迭代之间**没有数据依赖**（即 `c[i]` 的计算不依赖 `c[i-1]`）。编译器可以自动分析这一点（Auto-vectorization），也可以由程序员保证（通过 `restrict` 关键字或手写 intrinsics）。

**ML 编译器的处理**：Triton、TVM 等在 Tile 粒度上做向量化——先用 Loop Tiling 把数据切成适合 SRAM 的块，再对 Tile 内部进行向量化。

#### Loop Parallelization（循环并行化）

将循环迭代分配给多个线程（CPU 上的 OpenMP 线程，或 GPU 上的 CUDA Thread Block）：

```c
// 块分配（Block Distribution）：线程 i 处理连续的 items 个元素
int items = (N + num_threads - 1) / num_threads;
int start = thread_id * items;
int end   = min(start + items, N);
for(i = start; i < end; i++)
    c[i] = a[i] * b[2*i];

// 轮询分配（Round-Robin）：线程 i 处理 i, i+P, i+2P, ...（P = 线程数）
for(i = thread_id; i < N; i += num_threads)
    c[i] = a[i] * b[2*i];
```

**块分配** 具有更好的 Cache 局部性（每个线程访问连续内存）；**轮询分配** 在负载不均匀时更平衡。GPU 的 Thread Block 使用块分配逻辑。

#### Software Pipelining（软件流水）

当循环体内部存在**长依赖链**但**迭代间相互独立**时，通过展开并重排指令使不同迭代的操作重叠执行：

```c
// 原始循环（依赖链：a = 2*B → b = a+1 → c = b/2，3 步）
for(i = 0; i < N; i++) {
    a = 2 * B[i];
    b = a + 1;
    c = b / 2;
}

// 软件流水展开（展开因子 3，重排使不同迭代的操作并行）
for(i = 0; i < N/3; i += 3) {
    a  = 2 * B[i];    a1 = 2 * B[i+1];    a2 = 2 * B[i+2];
    b  = a  + 1;      b1 = a1 + 1;        b2 = a2 + 1;
    c  = b  / 2;      c1 = b1 / 2;        c2 = b2 / 2;
}
```

**适用场景**：

- **In-order 处理器**（如 GPU 的 CUDA Core）：软件流水能显著提升 IPC，因为处理器不能自己发现跨迭代的独立性
- **Out-of-order 处理器**：乱序执行硬件已经自动发现并行性，软件流水收益较小，但大展开因子仍有助于向量化

---

### 7.3 多循环变换

#### Loop Interchange（循环交换）

改变嵌套循环的遍历顺序：

```c
// 原始：i → j → k 顺序
for(i = 0; i < M; i++)
    for(j = 0; j < N; j++)
        for(k = 0; k < K; k++)
            C[i][j] += A[i][k] * B[k][j];

// 交换后：i → k → j 顺序（对 Row-major 存储更友好）
for(i = 0; i < M; i++)
    for(k = 0; k < K; k++)
        for(j = 0; j < N; j++)   // B[k][j] 现在是连续访问
            C[i][j] += A[i][k] * B[k][j];
```

**分析**（Row-major 存储）：

- `A[i][k]`：交换前 k 变化（连续），交换后 k 在中间层变化 → 仍连续。但现在 `A[i][k]` 在最内层不变化（可以提到循环外，复用）
- `B[k][j]`：交换前 j 变化（不连续，需要按列访问）→ 交换后 j 变化（连续！）
- `C[i][j]`：交换前 j 变化（连续），交换后 j 变化（仍连续）

Loop Interchange 的合法性条件：变换后的循环顺序必须不改变任何数据依赖（即如果 `C[i][j]` 的计算没有依赖前一次迭代的 `C[i][j]`，则合法）。

#### Loop Blocking / Tiling（循环分块）⭐ 最重要的变换

**动机**：矩阵乘法 $C = A \times B$ 中，若 $A$（$M \times K$）和 $B$（$K \times N$）都很大，则遍历时产生大量 Cache Miss。解决方法是将矩阵分成适合 Cache 的小块（Tile），每次只处理一个 Tile：

```c
// 矩阵转置的分块示例（原始版本 B[j][i] = A[i][j]）

// 原始（不分块）：B 的写入是按列跳跃的，Cache Miss 率高
for(i = 0; i < M; i++)
    for(j = 0; j < N; j++)
        B[j][i] = A[i][j];

// 分块后（Block Size = T）
for(i = 0; i < M; i += T)
    for(j = 0; j < N; j += T)          // 以 T×T 块为单位
        for(ii = i; ii < min(i+T, M); ii++)
            for(jj = j; jj < min(j+T, N); jj++)
                B[jj][ii] = A[ii][jj]; // 在块内，A 和 B 都有局部性
```

**为什么分块有效（图示）：**

假设缓存只能容纳 4 个 Cache Line（每行 2 元素），矩阵大小 8×8：

```
未分块访问 B[j][i]（列访问）：
A 的访问顺序（行）：A[0,0] A[0,1] A[0,2] ... A[0,7]  → 每次顺序读（好）
B 的访问顺序（列）：B[0,0] B[1,0] B[2,0] ... B[7,0]  → 每次跨行跳跃（坏，Cache Miss）

分块后（T=4），处理块 [i:i+4][j:j+4]：
每个块内，A 和 B 都只访问 4×4 个元素，可以完全放进 Cache，在块内的所有访问都命中
```

**Tile 大小的选择**：Tile 要足够小以放入目标 Cache 层（通常是 L1 或 L2），但又要足够大以摊销 Tile 加载开销。这是 Autotuning 的核心参数。

**Triton 的 Tile 抽象**：Triton 将 Tiling 作为核心编程原语，程序员直接以 Tile 为单位编写 kernel，编译器处理 Tile 内的细节（向量化、寄存器分配）：

```python
# Triton kernel：每个 program（类似 Thread Block）处理一个输出 Tile
@triton.jit
def matmul_kernel(A, B, C, M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    # 每个 program 计算 C 中 [pid_m*BLOCK_M:(pid_m+1)*BLOCK_M, pid_n*BLOCK_N:(pid_n+1)*BLOCK_N] 的 Tile
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(A + ...)   # 加载 A 的 Tile
        b = tl.load(B + ...)   # 加载 B 的 Tile
        acc += tl.dot(a, b)    # Tile 内的矩阵乘
    tl.store(C + ..., acc)
```

#### Loop Fusion（循环融合）

将两个独立的循环合并为一个，减少中间结果的内存写入和读取：

```c
// 未融合：两次遍历，中间结果 A[i] 需要完整写入内存
for(i = 0; i < N; i++)   A[i] = B[i];        // 写 A
for(i = 0; i < N; i++)   A[i] = 2 * A[i];    // 读 A，写 A

// 融合后：单次遍历，中间结果保留在寄存器
for(i = 0; i < N; i++) {
    A[i] = B[i];
    A[i] = 2 * A[i];   // A[i] 在寄存器中，不需要写回再读
}
```

**与算子融合的关系**：图级别的算子融合（Module 6）最终在循环层面就是 Loop Fusion。FlashAttention 的实现就是将 QK^T 计算循环和 Softmax 计算循环融合，使中间结果 N×N 的注意力矩阵全程在 Shared Memory 而不落入 HBM。

#### 其他重要变换

**Loop Unroll-and-Jam（展开并合并）：** 将外层循环展开，然后与内层循环融合，可以产生更大的向量化机会。

**Loop Skewing（循环倾斜）：** 通过修改循环下界使原本有依赖的波前（Wavefront）并行变得可能（常用于 Stencil 计算和 DP 算法）。

---

### 7.4 Autotuning（自动调优）

#### 参数空间与最优性

所有 Loop 变换都有关键参数：

- 展开因子（Unroll Factor）
- 分块大小（Tile Size）：可能是 $M$、$N$、$K$ 三个维度各自的大小
- 融合策略（哪些循环融合）
- 循环顺序（Loop Interchange 后的顺序）

**这些参数没有精确理论公式**——最优参数取决于：程序本身 × 机器硬件（Cache 大小、向量宽度、功能单元数量）× 输入数据规模。这就需要 **Autotuning（自动调优）**。

#### 搜索策略

**暴力搜索（Brute Force）：**

```python
best_time = inf
for params in parameter_space:   # 枚举所有参数组合
    compile(program, params)
    time = measure(program)
    if time < best_time:
        best_time = time
        best_params = params
```

参数空间通常是组合爆炸（多维 Tile Size 各有几十个候选），完全暴力不可行。

**稀疏搜索（Sparse Search）：** 只在有限的代表性参数集合上搜索。PyTorch 内置的 cuBLAS Autotuning 就是这个思路——预先定义几个候选 Tile 配置，运行时选最快的。

**启发式搜索（Heuristic Search）：**

```python
while budget_remaining():
    params = heuristic.suggest(history)  # 用历史数据建议下一个尝试的参数
    time   = compile_and_measure(program, params)
    heuristic.update(params, time)
    if time < best_time:
        best_time, best_params = time, params
```

常用启发式算法：进化算法（Evolutionary Algorithm）、模拟退火（Simulated Annealing）、贝叶斯优化（Bayesian Optimization）。Apache TVM 的 AutoTVM / MetaSchedule 使用这类方法。

#### ML 框架的 Autotuning 工具链

以下是 Loop 优化与 Autotuning 的工具链谱系，从低层到高层：

```
低层（手写代码 + 优化器）
├── 手写 CUDA + IREE / PLUTO 自动优化
│
中层（DSL + Schedule + Autotuning）
├── Halide（图像处理，将算法与 Schedule 分离）
└── Apache TVM（通用 ML 编译器，AutoTVM / MetaSchedule）
│
高层（Tiled DSL + Autotuning）
├── OpenAI Triton（GPU，以 Tile 为基本抽象，自动处理低层细节）
├── Google Pallas（TPU/GPU）
└── NVIDIA cuTile / Tilus（NVIDIA 内部，较新）
```

**Triton 的特殊地位**：Triton 是目前 ML Kernel 开发的实际标准工具。它让程序员只需要描述 Tile 级别的逻辑（不需要手写 Warp/Thread 调度、Shared Memory 管理），编译器自动完成这些优化，同时保持接近手写 CUDA 的性能。PyTorch 2.0 的 `torch.compile` 后端（TorchInductor）就是通过 Triton 生成 GPU kernel 的。

---

## Module 8：执行 ML 程序与 Roofline 模型

### 8.1 宏观调度：专用加速器

考虑一个简单神经网络的算子图：

```
x → conv1 → relu → conv2 → relu → conv3 → relu → conv4 → pixel_shuffle
```

**最简单的加速器设计**：为每种算子类型建一个专用功能单元，各单元有输入/输出缓冲区：

```
Conv Unit │ ReLU Unit │ PixelShuffle Unit
   ↑↓           ↑↓              ↑↓
         共享数据总线 / 控制器
```

**串行执行时间线**（各单元轮流使用）：

```
时间 →
Conv │ ReLU │ Conv │ ReLU │ Conv │ ReLU │ Conv │ PixSh │
```

**Pipeline Parallelism（流水线并行）** 的引入：

```
时间 →
Conv1 [Input1] │ Conv1 [Input2] │ Conv1 [Input3]
               │ ReLU1 [Input1] │ ReLU1 [Input2]
                                │ Conv2 [Input1]
```

不同的功能单元同时处理不同输入的不同层。吞吐量提升（$\approx$ 单元数倍），但代价是：

1. 需要更多缓冲区（每个单元都要能独立缓存自己的输入）
2. 需要更多计算单元副本（不能让 Conv1 和 Conv2 共享一个 Conv 单元）
3. Pipeline 效率受限于最慢的那个单元（木桶效应）

**Data Parallelism** 进一步扩展：为每个输入复制完整的硬件路径，多输入完全并行处理。

> **关键教训**：增加并行度（无论是 Pipeline 还是 Data Parallel）总是需要增加资源——内存和计算。这是 ML 系统设计的基本权衡。

### 8.2 宏观调度：通用处理器

在通用 GPU/CPU 上执行 ML 图面临"一个尺寸无法满足所有需求"的挑战：

- 共享计算资源（所有算子竞争同一批 CUDA Core）
- 共享内存（所有算子读写同一块 HBM）
- 所有算子以软件实现

#### 动态调度（Dynamic Scheduling）

```
Scheduler
    ↓
算子队列 [matmul, conv, relu, ...]
    ↓ 分割成 Worker 大小的工作块
Global Work Queue
    ↓ Worker Threads 抢取执行
[Worker 0] [Worker 1] [Worker 2] ...
```

这是 PyTorch（以及大多数 ML 框架）的基本执行模型。优点：灵活，支持动态控制流和动态形状。缺点：调度开销，难以精确预测执行时序。

独立算子（在 DAG 中无相互依赖的）可以在资源允许时并发执行——这是 **Inter-Operator Parallelism** 的来源。

#### 静态调度（Static Scheduling）

**前提**：如果所有输入输出的大小固定（非动态），则每个算子的运行时间可以在执行前预测。根据这个时序信息和数据流，提前生成确定性的调度方案（类似硬件流水线的工作方式）。

这种方式目前主要见于：

- 专用芯片（Groq 的 TSP 处理器强制要求静态调度）
- Google TPU 的某些执行模式（XLA 编译器）

#### 宏观图优化汇总

|优化技术|作用|适用条件|
|---|---|---|
|**算子融合**|减少中间内存读写|相邻算子在 DAG 上连续|
|**图分割（Partitioning）**|将图分配到不同设备|模型过大放不进单 GPU|
|**Data Parallelism**|多输入并发，不同 Copy 处理不同 Batch|Batch 可分割|
|**Pipeline Parallelism**|不同层在不同设备，流水处理|模型层次深，层间通信可承受|
|**Tensor Parallelism**|单层内矩阵跨设备分割|单层过大或追求延迟|

### 8.3 微观内核优化

给定一个算子（如矩阵乘法），如何编写高性能的 Kernel？这是"显微视角"（Microscopic View）的问题。

#### Little's Law 在内核优化中的应用

**Little's Law**：$n = R \times t$，其中 $n$ 是系统中同时在途的操作数，$R$ 是吞吐率（操作/周期），$t$ 是每次操作的延迟（周期）。

**应用到计算吞吐**：假设一个 GPU 的 FMA 单元每周期可以接受 2 条 FMA 指令，每条 FMA 有 4 周期延迟，则要**充分利用**该 FMA 单元，需要同时 in-flight 的 FMA 数量为：

$$n = R \times t = 2 \times 4 = 8 \text{ 条 FMA}$$

每条 FMA 需要 3 个操作数寄存器，8 条 FMA 同时 in-flight 需要 $8 \times 3 = 24$ 个寄存器——这决定了 kernel 的**最低寄存器需求**。

**应用到内存带宽**：假设 HBM 带宽为 1 load/周期（64 bits），Load 延迟为 100 周期，则要饱和带宽需要：

$$n = 1 \times 100 = 100 \text{ 个 in-flight Load}$$

实际中，一次 Cache Line 传输（64 bytes ≈ 8 × 64-bit loads）可以视为一次请求，所以大约需要 ~100/8 ≈ 12–13 个 in-flight Cache Line 请求。

**延迟隐藏（Latency Hiding through Parallelism）**：

核心思想：$n$ 个并发操作可以"隐藏"每次操作的延迟 $t$。若并发度不够（$n < R \times t$），就会出现流水线停顿，吞吐率下降：

```
足够并发（n ≥ R×t）：
[FMA0][FMA1][FMA2][FMA3] ... ← 始终有新 FMA 可以发射
       [FMA0写回]              ← FMA0 完成时，FMA3 已在执行

并发不足（n < R×t）：
[FMA0][ 等待 ][ 等待 ][ 等待 ][FMA0写回][FMA1]...
            ← 流水线停顿，FMA 单元空转
```

这正是 GPU 在一个 SM 上调度 2048 个线程的原因：当一个 Warp 等待内存时，另一个 Warp 立即填上，保持计算单元忙碌。

### 8.4 Roofline 模型

Roofline 模型是判断程序是否达到硬件性能上限的核心工具，也是你的 Quantization Profiler 项目的理论基础。

#### 算术强度（Arithmetic Intensity）

$$I = \frac{\text{FLOPs}}{\text{Bytes}}$$

也记为 $I$ (FLOPs/Byte)：每从内存读入 1 字节数据所对应的浮点运算次数。

**注意**：这里的"Bytes"是**内存流量（Memory Traffic）**，不是数据大小，需要考虑数据复用（数据从 HBM 载入 Cache 只算一次，后续对该数据的访问不算）。

**常见算子的算术强度（参考值）：**

|算子|典型 $I$ |瓶颈类型|
|---|---|---|
|向量加法 $y = a + b$ |0.5 FLOP/Byte（FP32）|Memory-Bound|
|ReLU|0.25 FLOP/Byte|Memory-Bound|
|Softmax|~1 FLOP/Byte|Memory-Bound|
|LayerNorm|~2 FLOP/Byte|Memory-Bound|
|小矩阵乘法（M=N=K=64）|~32 FLOP/Byte|Memory-Bound|
|大矩阵乘法（M=N=K=1024）|~512 FLOP/Byte|Compute-Bound|
|注意力（序列长度 512）|~50 FLOP/Byte|Compute-Bound|

#### Roofline 公式

$$P = \min\left(T_{peak},\ \beta \times I\right) \text{ FLOP/s}$$

其中：

- $T_{peak}$：处理器的**峰值计算吞吐**（TFLOP/s）
- $\beta$：处理器的**内存带宽**（GB/s 或 TB/s）
- $I$：程序的**算术强度**（FLOP/Byte）
- $P$：程序可达到的**性能上界**（TFLOP/s）

**Roofline 图（手绘理解）：**

```
性能 P
(TFLOP/s)  │              ╔══════════════════ T_peak（计算天花板）
           │           ╔═╝
           │        ╔═╝         ← 斜率 = β（内存带宽）
           │     ╔═╝
           │  ╔═╝     △ 这段：Memory-Bound
           │╔═╝             ← 这个折点：Ridge Point（脊点）
           └────────────────────────────→ I (FLOP/Byte)
                          △ 这段：Compute-Bound
```

**脊点（Ridge Point）**：$I_{ridge} = T_{peak} / \beta$。算术强度低于 $I_{ridge}$ 的程序受内存带宽限制；高于 $I_{ridge}$ 的程序受计算能力限制。

**NVIDIA A100 SXM 的参考数据：**

- $T_{peak}$ (FP16 Tensor Core) ≈ 312 TFLOP/s
- $\beta$ (HBM2e) ≈ 2 TB/s
- $I_{ridge}$ = 312 / 2000 = 156 FLOP/Byte

即对于 A100，**算术强度超过 156 FLOP/Byte 才能进入 Compute-Bound 区间**。大多数 LLM 推理中的 MatMul（由于 batch size 小），实际算术强度远低于此——这正是 LLM 推理难以充分利用 GPU 算力的根本原因，也是 Quantization 和 Flash Decoding 等优化的动机。

**RTX 4060 的参考数据（你的 Profiler 使用的 GPU）：**

- $T_{peak}$ (FP16) ≈ 33 TFLOP/s
- $\beta$ (GDDR6) ≈ 272 GB/s
- $I_{ridge}$ = 33,000 / 272 ≈ 121 FLOP/Byte

#### Roofline 在实践中的应用

**如何测量一个 kernel 的 $I$：**

1. 使用 `nsight compute` 或 `rocprof` 统计实际 HBM 流量（bytes）
2. 使用性能计数器统计实际 FP 运算数（或理论计算）
3. $I = \text{FLOPs} / \text{HBM bytes}$

**根据 Roofline 确定优化方向：**

|程序落点|瓶颈|优化方向|
|---|---|---|
|远低于两条 Roofline|实现低效|检查 Warp 利用率、Branch Divergence、Load/Store 效率|
|靠近带宽 Roofline，低于 Ridge|Memory-Bound|增加数据复用（Tiling）、使用低精度（FP16→INT8）、算子融合|
|靠近计算 Roofline，高于 Ridge|Compute-Bound|使用 Tensor Core（FP16/INT8）、增大 batch size|
|已达 Roofline|接近最优|无需继续优化（或换更好的硬件）|

**量化对 Roofline 的影响（直接关联你的 Profiler 项目）：**

以 INT4 量化为例：

- 权重从 FP16 变为 INT4：内存流量减少 4×（相同 Byte 数现在携带 4× 更多权重）
- Bytes 减少 4×，FLOPs 不变 → $I$ 增大 4×
- 原本 Memory-Bound 的小 batch 推理，量化后 $I$ 越过 Ridge Point，变为 Compute-Bound
- 但如果 INT4 计算在 Tensor Core 上的 $T_{peak}$ 也提高（INT4 Tensor Core 比 FP16 快 4×），则天花板也同步提升

Marlin（Sparse INT4 GEMM Kernel）的 batch-size 分界行为，就是这个 Roofline 效应的实例：小 batch 时 Memory-Bound，随 batch size 增大算术强度提升，逐渐进入 Compute-Bound 区间。

---

## Module 9：ML 程序的内存与存储

### 9.1 内存分配基础

#### 程序的内存来源

**静态分配（Static Allocation）**：全局变量、静态数组，在编译时确定大小，程序启动时分配，生命周期贯穿程序运行。

**动态分配（Dynamic Allocation）**：`malloc` / `free`（C）或语言运行时 GC，在堆（Heap）上按需分配。

**GPU 设备内存**：需要使用设备特定的分配器：

```c
// CPU 内存
float *cpu_ptr = malloc(N * sizeof(float));
free(cpu_ptr);

// GPU 内存
float *gpu_ptr;
cudaMalloc(&gpu_ptr, N * sizeof(float));
cudaFree(gpu_ptr);
```

**注意**：`cudaMalloc` 返回的指针值看起来和 CPU 指针没有区别（同一 64 位地址空间），但在 CPU 代码中直接解引用 GPU 指针（或在 GPU 中解引用 CPU 指针）会导致 Segmentation Fault（无 UVA 时）或极慢的 PCIe 访问（有 UVA 时）。

#### GPU 的多种内存空间（CUDA 命名）

|内存空间|位置|特点|用途|
|---|---|---|---|
|**Global Memory**|HBM（片外）|大（GB 级），慢（~600 周期延迟），对所有线程可见|主要数据存储|
|**Shared Memory**|片内 SRAM|小（每 SM 128KB），快（~几十周期），仅 Block 内线程共享|Tile 缓存，Bank Conflict 需注意|
|**Register**|寄存器文件|最快，每 SM 有限（65536 个 32-bit register）|局部变量，中间计算结果|
|**Constant Memory**|HBM + 专用 Cache|只读，有广播优化（所有线程读同一地址时只一次 HBM 访问）|卷积滤波器、常量张量|
|**Local Memory**|实际是 HBM，逻辑上 per-thread|用于溢出的寄存器（Register Spilling）|避免使用（性能差）|

某些加速器（如 Google TPU、Pixel TPU）没有动态内存分配能力，所有内存布局必须在编译时静态确定——这是它们适合静态调度的原因，也限制了它们处理动态形状的能力。

#### 动态内存分配的性能问题

在以下情况下，内存分配本身会成为性能瓶颈：

1. **高频率分配/释放**：每次 `cudaMalloc` 在 GPU 上的开销极大（涉及设备同步），不能在每次前向传播时分配激活值内存
2. **内存不足（Memory Pressure）**：当可用内存池接近耗尽，分配器花大量时间寻找合适的空闲块
3. **碎片化（Fragmentation）**：大量小分配后，内存中虽有足够总空间但没有连续大块

**解决方案**：

- **内存池（Memory Pool）**：PyTorch 的 `caching_allocator` 就是这个机制——分配的内存不立即归还给 CUDA，而是缓存在池中供下次使用，避免频繁的 `cudaMalloc` / `cudaFree`
- **内存规划（Memory Planning）**：在执行前（编译时）分析整个计算图，规划每个张量的生存期（Liveness），复用不再活跃的张量所占内存（如 Activation 在 Backprop 完成后就可以释放）

### 9.2 ML 程序的内存使用

#### 内存使用的四类来源

在训练一个神经网络时，GPU 内存被以下四类数据占用：

|类别|描述|大小|可否复用|
|---|---|---|---|
|**Weights（权重）**|模型参数，推理时只读|固定，与模型大小正比|所有输入共享|
|**Activations（激活值）**|前向传播中各层的中间输出，反向传播时需要|与 Batch Size × 模型层数正比|可重计算（以计算换内存）|
|**Gradients（梯度）**|反向传播中每个参数的梯度|与 Weights 大小相同|每步更新后可释放|
|**Optimizer State（优化器状态）**|Adam 的一阶矩 $m$ 和二阶矩 $v$ |**2× 模型大小**（FP32）|每步更新后可释放（不能丢弃）|

**内存估算公式（Adam 优化器，FP32 训练）：**

$$M_{total} = \underbrace{P \times 4}_{\text{Weights}} + \underbrace{P \times 4}_{\text{Gradients}} + \underbrace{2P \times 4}_{\text{Adam: }m,v} + M_{act}$$

其中 $P$ 是参数量（个），4 是 FP32 每参数字节数。**仅模型状态就需要 16 bytes/参数**。175B 参数的 GPT-3：$175 \times 10^9 \times 16 \approx 2.8 \text{ TB}$ ——远超单个 GPU 的内存容量。

**典型模型参数量参考：**

|模型|参数量|FP16 权重大小|
|---|---|---|
|AlexNet|60M|~120 MB|
|GPT-3|175B|~350 GB|
|DeepSeek-V3|671B（MoE，激活 37B）|~1.3 TB（全量）|

#### 激活值内存的特殊问题

训练时，前向传播的每层输出（Activation）必须保留到反向传播使用完毕，这导致内存占用随模型深度线性增长。

**Activation Checkpointing（激活检查点）/ Gradient Checkpointing：**

策略：不保存所有层的激活值，只保存关键检查点层；在反向传播需要某层激活时，**从最近的检查点重新计算（Recomputation / Rematerialization）**。

```
前向传播（存所有激活，内存高）：
L0→[A0]→L1→[A1]→L2→[A2]→L3→[A3]→损失

使用 Checkpointing（只存 A0, A2，反向时重计算 A1, A3）：
L0→[A0]→L1→[    ]→L2→[A2]→L3→[    ]→损失
                                 ↑ 反向传播到这里时重新 L2→A3
                      ↑ 反向传播到这里时重新 L1→A1
```

**权衡**：节省约 $\sqrt{n}$ 倍内存（$n$ 为层数），代价是前向传播运行约 1.33× 时间（额外重计算约 1/3 的层）。

### 9.3 内存优化技术

#### 量化（Quantization）

将模型参数从高精度浮点转为低精度整数，直接减少内存占用：

|转换|内存节省|精度影响|典型用途|
|---|---|---|---|
|FP32 → FP16 / BF16|2×|极小|训练（混合精度）|
|FP32 → INT8|4×|小（需细心标定）|推理|
|FP32 → INT4|8×|中（需分组量化）|大模型推理（GPTQ、Marlin）|
|FP32 → INT2 / Binary|16×+|较大|实验性质|

**权重量化（Weight-only Quantization）**：只量化权重，激活值保持高精度。激活时反量化（Dequantize）后做高精度矩阵乘。主要节省 **内存带宽**（而非计算量），适合 Memory-Bound 的 LLM 推理。

**权重 + 激活量化（Full Quantization）**：权重和激活都量化，让整个矩阵乘法在 INT8/INT4 的 Tensor Core 上运行，同时节省 **内存** 和 **计算**。需要标定（Calibration）激活分布的动态范围。

#### 稀疏化（Sparsification）

将权重中绝对值接近零的参数强制置零，然后用稀疏格式存储：

**非结构化稀疏（Unstructured Sparsity）：**

- 稀疏位置完全随机
- 存储：COO / CSR 格式（见 Part I Module 4.5）
- 问题：随机 Gather/Scatter 访问，GPU 向量化效率低

**半结构化稀疏（Semi-structured / N:M Sparsity）：**

```
NVIDIA Ampere 的 2:4 稀疏性（每 4 个连续权重中有 2 个非零）：
原始: [0.5  0.0  0.3  0.0  0.7  0.2  0.0  0.1]
2:4:  [0.5  0.3  0.7  0.2]  +  [0  2  4  5]（索引，2bits/元素）
```

NVIDIA 的 cuSPARSELt 库支持 2:4 稀疏权重的高效矩阵乘法（使用 Ampere 的 Sparse Tensor Core），理论上比稠密矩阵快 2×（相同有效参数量的情况下）。

#### 压缩（Compression）

- **无损压缩**：对激活值/梯度的传输做熵编码（如 LZ4、Zstd），减少 PCIe 或 NVLink 传输量，但需要 CPU/GPU 时间解压
- **有损压缩**：即量化，或 QMoE 等极端压缩（Sub-1-Bit）

#### 权重文件的高效加载

加载大模型权重是 LLM 服务的重要瓶颈（GPT-3 175B FP16 = 350GB）：

**传统方式（PyTorch Pickle 文件）：**

1. Python Pickle 解析文件头（CPU 时间开销）
2. `torch.load()` 将数据复制进 CPU RAM
3. `.to(device)` 再复制到 GPU

** `mmap` 方式（接近最优）：**

```c
// 将权重文件直接映射进进程地址空间
FILE *f = fopen("/weights/llama-70b.bin", "r");
fseek(f, 0, SEEK_END);
size_t sz = ftell(f);
void *weights = mmap(NULL, sz, PROT_READ, MAP_SHARED, fileno(f), 0);
fclose(f);
// weights 指向的内存由 OS 按需从磁盘分页加载（不全部读入 RAM）
```

`mmap` 将文件视为虚拟内存（Page Fault 时懒加载），可以让模型权重分散在 CPU RAM 和 Disk 之间，仅加载当前需要的层（**模型并行推理中按层加载** 的基础技术）。

**Safetensors**（Hugging Face 开发）是目前推荐的现代权重格式，设计上支持 `mmap` 直接加载，避免 Pickle 的安全问题和性能开销。

---

## Module 10：通信——批量数据传输

### 10.1 单机内通信

#### 数据传输的三种方式

**Load/Store 指令**：CPU 通过 Load/Store 直接访问数据。灵活但占用 CPU 执行资源。适合小量数据。

**Non-temporal Load/Store**：特殊变体，提示 CPU 不要将这些数据缓存（适合大块只写一次的数据，避免污染 Cache）：

```c
// x86 的 non-temporal store（Streaming Store）
_mm_stream_si32(ptr, value);  // 直接写到内存，绕过 Cache
```

**DMA（Direct Memory Access）**：专用硬件引擎异步执行批量数据传输，CPU 只需启动（指定源、目标、大小），然后可以继续执行其他代码，传输完成后收到中断通知：

```c
// 概念性 DMA 启动（实际 GPU 使用 cudaMemcpyAsync）
cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream);
// 这行代码立即返回，数据传输在后台异步进行
// CPU（或另一个 CUDA Stream）可以立即做别的事
```

#### 计算与通信重叠（Compute-Communication Overlap）

**通信时间不可消除，但可以隐藏**。关键技术是**重叠（Overlap）**：在传输数据的同时执行其他计算。

```
无重叠（串行）：
[传输 Batch1 到 GPU][计算 Batch1][传输 Batch2 到 GPU][计算 Batch2]

有重叠（流水）：
[传输 Batch1][计算 Batch1]
             [传输 Batch2][计算 Batch2]
                          [传输 Batch3][计算 Batch3]
```

**CUDA Stream** 是实现这种重叠的机制：不同 Stream 上的操作可以并发执行（当资源允许时）。

#### Pinned Memory（锁定内存）

DMA 引擎传统上使用**物理地址**操作（而非虚拟地址），因为 DMA 在 CPU 的 MMU 之外工作。但虚拟页可能被 OS 换出（Swap），导致 DMA 访问到错误的物理地址。

**Pinned Memory（锁定内存 / Page-locked Memory）**：通过 `cudaHostAlloc()` 或 `mlock()` 分配的内存，OS 保证该内存页不会被换出，从而允许 DMA 安全使用其物理地址。

**收益**：锁定内存的 H2D/D2H 传输速度约是普通内存（Pageable Memory）的 2–3×，因为省去了先复制到内核缓冲区（Bounce Buffer）的步骤：

```
普通内存 H2D 传输路径：
CPU User Buffer → (CPU-to-CPU copy) → Kernel Bounce Buffer → DMA → GPU

锁定内存 H2D 传输路径：
CPU Pinned Buffer → DMA（直接）→ GPU
```

PyTorch 中 `DataLoader(pin_memory=True)` 就是启用这个优化。

#### GPUDirect 技术

**GPUDirect RDMA**：允许 GPU 直接访问网卡（NIC）缓冲区，绕过 CPU 内存，大幅减少延迟和 CPU 负担：

```
传统路径：GPU →（PCIe）→ CPU RAM →（PCIe）→ NIC → 网络
GPUDirect：GPU →（PCIe + P2P DMA）→ NIC → 网络
```

**GPUDirect P2P（Peer-to-Peer）**：同一机器上的 GPU 之间直接通过 NVLink 传输数据，绕过 CPU 内存。

### 10.2 网络通信基础

#### 网络拓扑

分布式 ML 训练中，机器间的网络连接方式（拓扑）直接影响 AllReduce 等集体通信的效率：

|拓扑|特点|典型应用|
|---|---|---|
|**Ring**|每个节点连接左右两个邻居；直径大（N/2），但 Bisection Bandwidth 稳定|Ring AllReduce（带宽效率最优）|
|**Tree**|层次结构，传播快（logN 跳）；根节点是瓶颈|Broadcast|
|**Fat-Tree（胖树）**|上层交换机有更多端口；无带宽瓶颈，但结构复杂、成本高|数据中心主流（InfiniBand 网络）|
|**2D/3D Torus**|每个节点连接 2/3 个维度的邻居；高 Bisection Bandwidth|Google TPU Pod 互联|
|**Fully Connected**|每对节点直接相连；成本 $O(N^2)$ |单机 NVLink（A100 NVLink Switch）|

**关键度量指标：**

- **Bisection Bandwidth**：将网络二等分所需切断的最少带宽总和。Bisection Bandwidth 越高，大规模 AllReduce 效率越好。
- **Diameter**：网络中最长的最短路径。Diameter 越小，点对点延迟越低。

**NVIDIA A100 NVLink 拓扑（单机 8 GPU）：**

每个 A100 通过 NVLink 4.0 与其他 7 块 GPU 连接（接近全连接），单向带宽约 600 GB/s，是 PCIe 5.0 的约 10×。这使得单机 8 GPU 的 AllReduce 可以以近带宽峰值运行。

#### 通信协议栈简述

从底到顶：

```
应用层（HTTP, gRPC, NCCL, MPI）
    ↕
传输层（TCP, SCTP, RDMA Verbs）
    ↕
网络层（IPv4, IPv6, InfiniBand LID）
    ↕
数据链路层（Ethernet, InfiniBand）
    ↕
物理层（铜缆, 光纤, 电磁波）
```

**ML 训练关注的是最底层和最顶层**：

- 底层：选择 InfiniBand 还是 RoCE（RDMA over Converged Ethernet）影响延迟和带宽
- 顶层：NCCL 等库在 OS 不介入的情况下直接操作网卡（User-space Networking），减少延迟

**TCP vs RDMA 的区别：**

|特性|TCP|RDMA|
|---|---|---|
|延迟|高（内核参与，~μs 级）|低（绕过内核，~ns-μs 级）|
|CPU 开销|高（CPU 执行 copy）|低（零拷贝，DMA）|
|可靠性|有（ACK 机制）|有（InfiniBand）或无（RoCE 需要 PFC）|
|适用|通用网络|高性能计算/ML 训练|

### 10.3 Collective Communication 算法

集体通信（Collective Communication，CC）是所有参与设备共同执行的多步数据传输+计算操作。

#### AllReduce：分布式训练的核心原语

**问题**：$P$ 个 GPU 各持有梯度 $g_i$，需要计算全局平均梯度 $g = \frac{1}{P}\sum_i g_i$ 并让每个 GPU 都获得结果。

**朴素实现（汇聚求和）：**

```python
# 每个 GPU 发送数据给 GPU 0，GPU 0 求和后广播
# 通信量：O(P × |g|)，GPU 0 是瓶颈
for m in other_devices:
    send(m, my_gradient)
result = my_gradient
for m in other_devices:
    result += recv(m)
broadcast(result)
```

**Ring AllReduce（带宽最优）：**

Ring AllReduce 是目前 NCCL 和 PyTorch DDP 使用的标准算法，分两个阶段：

**阶段一：Reduce-Scatter**（每个 GPU 收到所有 GPU 对应自己负责的那 1/P 段数据的 sum）

```
每个 GPU 的数据分成 P 段：[S0|S1|S2|S3]（P=4 为例）

轮次 1：GPU_i 向 GPU_{i+1} 发送 S_i 的副本，同时接收 S_{i-1} 并累加
轮次 2：GPU_i 向 GPU_{i+1} 发送累加后的 S_{i-1}，同时接收并累加 S_{i-2}
轮次 3：类似...

经过 P-1 轮后，GPU_i 有 S_i 的全局 sum
```

**阶段二：AllGather**（每个 GPU 广播自己负责的那段，所有 GPU 收集完整 reduce 结果）

```
每个 GPU 将自己的 S_i（已是全局 sum）发给下一个 GPU
经过 P-1 轮后，每个 GPU 都有完整的 [S0+S1+S2+S3]
```

**Ring AllReduce 的通信量分析**：每个 GPU 发送和接收 $2 \times \frac{P-1}{P} \times |g|$ 数据，当 $P$ 较大时趋近于 $2|g|$，与 GPU 数量无关——这是 Ring AllReduce 带宽效率最优的原因。

#### 其他 Collective Communication 算法

|算法|语义|通信方向|场景|
|---|---|---|---|
|**Broadcast**|Root 发送数据给所有节点|1 → N|参数服务器广播模型权重|
|**Reduce**|所有节点的数据聚合到 Root|N → 1|（较少用，被 AllReduce 替代）|
|**AllReduce**|所有节点聚合，结果广播给所有人|N → N|**梯度同步**（DDP）|
|**Gather**|所有节点的数据发给 Root|N → 1|收集推理结果|
|**AllGather**|所有节点的数据广播给所有人|N → N|**FSDP 的参数重建**|
|**Scatter**|Root 将数据分发给各节点|1 → N|分发 Batch 数据|
|**ReduceScatter**|AllReduce 的前半段|N → N（每人只保留一部分）|**FSDP 的梯度 Shard**|
|**AlltoAll**|每个节点发送不同数据给每个其他节点|N → N（全对全）|**MoE 的 Expert 路由**|
|**Barrier**|同步点，所有节点等待彼此到达|—|迭代间同步|

#### 通信库的现状

|库|维护方|特点|
|---|---|---|
|**NCCL**|NVIDIA|GPU 间通信的事实标准；针对 NVLink 和 InfiniBand 优化|
|**RCCL**|AMD|基于 NCCL 的 AMD ROCm 版本|
|**MSCCL**|Microsoft|允许自定义通信算法（如 Synthesis-based 算法）|
|**MPI**|学术/HPC 生态|传统 HPC 标准，对 GPU 支持较弱|

**40% 的训练成本是通信成本**——这是 ML 系统领域的重要数据点，也是通信优化研究持续活跃的原因。

### 10.4 ML 程序中的通信

#### 不同并行策略对应的通信类型

|并行策略|通信发生时机|使用的 CC 算法|通信量|
|---|---|---|---|
|**Data Parallel（DDP）**|每步 Backward 后梯度同步|AllReduce| $O(\text{参数量})$ / 步|
|**Pipeline Parallel**|每层激活值前向传递 / 梯度后向传递|P2P Send/Recv| $O(\text{层输出大小})$ / 微批|
|**Tensor Parallel**|矩阵分块后合并结果|AllReduce（层内）| $O(\text{激活大小})$ / 层|
|**FSDP（ZeRO-3）**|每层前向时 AllGather 参数；Backward 时 ReduceScatter 梯度|AllGather + ReduceScatter| $O(\text{参数量})$ / 步（与 DDP 相同但分散到每层）|
|**Expert Parallel（MoE）**|Token 路由到 Expert + 结果收集|AlltoAll| $O(\text{batch 大小})$ / MoE 层|

#### ML 特有的通信优化

**梯度压缩（Gradient Compression）**：在 AllReduce 前压缩梯度（如 1-bit SGD、PowerSGD 低秩近似），减少通信量。代价：引入近似误差（通常用误差反馈机制补偿）。

**使用过期数据（Stale Synchronous Parallelism）**：不等所有 GPU 完成梯度计算就开始下一步，允许使用略过时的梯度。例如：

- **DistriFusion**（扩散模型分布式推理）：在不同 GPU 上并行去噪步骤，使用上一步的激活值近似替代当前步（步间激活值变化小），几乎不损精度但实现推理并行化
- **Hogwild!**（异步 SGD）：多个 Worker 不加锁地更新共享参数，允许写冲突（稀疏梯度下近似无损）

**通信与计算重叠**：PyTorch DDP 的 **Bucketed Gradient AllReduce**：将参数分组为"Bucket"，当一个 Bucket 的所有参数的梯度计算完毕，立即启动这个 Bucket 的 AllReduce，与后续层的反向传播**并发**进行：

```
反向传播：   [L_n Backward][L_{n-1} Backward][L_{n-2} Backward]
AllReduce：              [AllReduce B2]        [AllReduce B1]
                                              ↑ 重叠！
```

---

## Module 11：Training I — 自动微分与梯度下降

### 11.1 训练流程概述

神经网络训练是一个**迭代优化过程**，每次迭代（Iteration / Step）包含：

```
初始化
   ↓（随机权重，不能全零）
┌──────────────────────────────────────┐
│ 1. Forward Pass                       │
│    输入 x → 计算预测 ŷ               │
│    计算损失 L = loss(ŷ, y)            │
│                                       │
│ 2. Backward Pass（反向传播）           │
│    2a. 计算梯度 ∂L/∂W（自动微分）     │
│    2b. 梯度下降：W ← W - η · ∂L/∂W  │
│                                       │
│ 3. 检查收敛（Loss 是否足够小？）      │
└──────────────────────────────────────┘
    │ 未收敛
    └──────→ 下一次迭代（repeat）
```

**为什么不能全零初始化**：若所有权重为零，每个神经元的输出和梯度都相同，所有权重按相同方式更新，网络永远无法打破对称性，等效于只有一个神经元。随机初始化打破对称性，使不同神经元学习不同的特征。

**前向传播（Forward Pass）与推理的区别**：训练时的前向传播需要**保存中间激活值**（供反向传播使用），而推理只需要最终输出。这是训练内存占用远高于推理的主要原因。

**收敛（Convergence）**：理论上不保证收敛，实践中通过：精心设计的初始化、适当的学习率调度（Learning Rate Schedule）、正则化（Regularization）等手段使其收敛。

### 11.2 自动微分（Automatic Differentiation）

#### 三种微分方法对比

|方法|原理|优点|缺点|
|---|---|---|---|
|**数值微分**| $\frac{\partial f}{\partial x} \approx \frac{f(x+\epsilon)-f(x)}{\epsilon}$ |实现简单|数值误差（截断 + 舍入），每参数需 2 次前向|
|**符号微分**|代数变换规则求导（如 Mathematica）|精确|表达式爆炸（Derivative Explosion）|
|**自动微分（AD）**|将程序拆解为基本操作，逐步应用链式法则|精确 + 高效|实现复杂（尤其是 Reverse AD）|

**现代 ML 框架（PyTorch、JAX、TensorFlow）全部使用自动微分。**

#### 链式法则（Chain Rule）

设 $y = f(g(h(x)))$，令 $w_0 = x, w_1 = h(w_0), w_2 = g(w_1), w_3 = f(w_2) = y$，则：

$$\frac{\partial y}{\partial x} = \frac{\partial y}{\partial w_2} \cdot \frac{\partial w_2}{\partial w_1} \cdot \frac{\partial w_1}{\partial x}$$

AD 的本质就是沿计算图应用链式法则，核心差异在于方向（前向 vs 后向）。

#### Forward Mode AD（前向模式）

使用**对偶数（Dual Numbers）** 同时传播原值和导数：

$$\tilde{x} = (x, \dot{x})$$

其中 $x$ 是实际值，$\dot{x}$ 是对某个输入变量 $x_i$ 的导数（初始化为 $\dot{x}_i = 1$，其他 $\dot{x}_j = 0$）。基本运算扩展为同时操作两个分量：

$$\tilde{a} + \tilde{b} = (a+b,\ \dot{a}+\dot{b})$$

$$\tilde{a} \times \tilde{b} = (ab,\ a\dot{b}+b\dot{a})$$

**执行**：将原始程序中的每个操作替换为对偶数操作，单次前向传播同时得到函数值和对某输入的梯度。

**缺点**：每次只能计算对**一个输入变量**的梯度。神经网络有 $n$ 个参数（通常百亿级），需要 $n$ 次前向传播——代价不可接受。

#### Reverse Mode AD（反向模式）⭐ 实际使用

**原理**：先完整执行前向传播，将所有中间值记录在**"Tape"（磁带）**上，然后**从输出反向**遍历计算图，依次计算梯度。

```
前向传播（记录中间值）：
x → [w1 = h(x)] → [w2 = g(w1)] → [w3 = f(w2)] = y
    ↑ 记录 w1         ↑ 记录 w2    ↑ 记录 w3

反向传播（从输出往回）：
dy/dy = 1
dy/dw2 = f'(w2) × 1 = f'(w2)
dy/dw1 = g'(w1) × f'(w2)
dy/dx  = h'(x) × g'(w1) × f'(w2)     ← 一次后向就得到对所有参数的梯度！
```

**关键优势**：**一次反向传播**就能得到对**所有参数**的梯度（$O(1)$ 次传播，与参数量无关）。这是为什么反向模式 AD 对 ML 训练至关重要。

**PyTorch 的实现（autograd）：**

```python
x = torch.randn(4, 4, requires_grad=True)
y = x @ x.T             # 记录操作到动态图（Tape）
z = y.relu()
loss = z.sum()
loss.backward()          # 触发反向传播，计算 loss 对所有 requires_grad 张量的梯度
print(x.grad)            # ∂loss/∂x
```

`requires_grad=True` 表示对该张量做计算时应记录到 Tape。`backward()` 触发反向遍历。

#### AD 的内存开销与优化

Reverse Mode AD 的主要代价是**存储所有前向传播中间值**（用于反向时计算局部梯度）。

**内存与计算的权衡策略：**

|策略|内存|计算|
|---|---|---|
|**完全存储**|最大（所有激活值）|最小（不重计算）|
|**完全重计算（No Storage）**|最小（只存输入）|最大（前向×2）|
|**Gradient Checkpointing**| $O(\sqrt{n})$（$n$ 层）|约 1.33× 原前向|

**Checkpointing 的数学原理**：将 $n$ 层分成 $\sqrt{n}$ 个段，每段存储一个检查点。反向传播到某段时，从最近的检查点重新计算该段的激活值。总存储 $\approx O(\sqrt{n})$，重计算开销 $\approx O(\sqrt{n})$ 次额外前向段。

PyTorch 接口：

```python
from torch.utils.checkpoint import checkpoint

def forward(x):
    # 只存 layer1 的输出，layer2 的中间值会在反向时重新计算
    x = layer1(x)
    x = checkpoint(layer2, x)   # layer2 的中间激活不存储
    x = checkpoint(layer3, x)
    return x
```

### 11.3 梯度下降与并行化

#### 梯度下降的变体

**全量梯度下降（Batch Gradient Descent）：**

$$W_{t+1} \leftarrow W_t - \frac{\eta}{N} \sum_{i=1}^{N} \nabla_W L_i$$

对整个训练集计算梯度的平均值，然后更新参数。方差小，但每步计算量大（需要遍历全数据集）。

**随机梯度下降（SGD）：**

$$W_{t+1} \leftarrow W_t - \eta \cdot \nabla_W L_i$$

每步只用**一个随机样本** $i$ 的梯度更新。计算快，但梯度有高方差（噪声大）。

**Mini-batch SGD（实际中使用）：**

$$W_{t+1} \leftarrow W_t - \frac{\eta}{B} \sum_{i \in \text{batch}} \nabla_W L_i$$

每步用**一小批（Batch）**样本的平均梯度。平衡了计算效率（可以 GPU 并行）和梯度质量（比单样本稳定）。

**带动量的 SGD / Adam 等优化器**：维护梯度的历史（一阶矩、二阶矩），自适应调整每个参数的学习率，收敛更快。代价是额外 2× 的优化器状态存储。

#### Hogwild!：无锁并行 SGD

**背景**：并行化 SGD 的自然想法是多个 Worker 同时计算梯度，但更新共享参数时需要锁，锁竞争严重损害性能。

**Hogwild! 的洞察**：若梯度是**稀疏的**（每次更新只影响参数的一小部分），不同 Worker 很少发生冲突（写入不同参数），因此即使**不加锁**地并发更新，实际的"写冲突"概率很低，对收敛影响很小。

**适用场景**：稀疏模型（如推荐系统的 Embedding 表）、稀疏输入（自然语言的词典）。不适合稠密梯度的模型（如 Transformer）。

### 11.4 分布式训练的存储与调度挑战

#### 训练规模的演变

|模型|年份|GPU 数量|数据集大小|
|---|---|---|---|
|AlexNet|2012|2× GTX 580|ImageNet ~130GB|
|GPT-2|2019|32× V100|40GB WebText|
|GPT-3|2020|~1024× A100（估计）|~570GB 文本|
|LLaMA-3 405B|2024|数千× H100|数十 TB|

#### Meta 的 Tectonic-Shift：大规模 ML 存储

**问题**：数百 TB 的训练数据、模型 Checkpoint（GPT-3 FP16 = 350GB，每小时一个检查点）存储需求远超单机 NVMe。

**Tectonic**（Meta 内部文件系统）的 ML 优化扩展 **Tectonic-Shift** 针对 ML 工作负载的特性做了定制：

- ML 训练读取数据是**顺序、重复的**（多 Epoch 遍历数据集）→ 预读（Prefetch）和缓存（Caching）优化
- Checkpoint 写入是**高带宽突发的**（同时写入千个 GPU 的状态）→ 写入缓冲和聚合
- 数据大小从 KB（小样本）到 TB（视频数据）不等 → 分层存储（SSD 缓存 + HDD 持久化）

#### Meta 的 MAST：全球 ML 训练调度

**问题**：跨越多个地理位置的数据中心（地理分布数据中心）调度 ML 训练作业，考虑：

- 不同数据中心的 GPU 可用性不同（故障、维护）
- 数据中心间网络带宽有限
- 作业中途的抢占（其他更高优先级作业需要资源）
- 局部 Checkpoint 恢复

**MAST 的核心思路**：全局调度器维护所有数据中心的资源状态，在作业提交时选择最优的数据中心组合，并在运行中动态迁移或重新调度以最大化 **Goodput（有效吞吐）**：

$$\text{Goodput} = \frac{\text{有效完成的训练步数}}{\text{时钟时间}}$$

有别于 **Throughput（吞吐）**（只考虑运算量），Goodput 排除了故障恢复、通信阻塞等无效时间。

---

## Module 12：Training II — 并行化方法全景

### 12.1 Data Parallelism（数据并行）

**基本设置：**

```
GPU 0: 完整模型副本 + Batch 的 1/P
GPU 1: 完整模型副本 + Batch 的 1/P
...
GPU P: 完整模型副本 + Batch 的 1/P
```

**训练步骤：**

1. 每个 GPU 独立做前向传播和反向传播，得到局部梯度 $g_i$
2. **AllReduce 梯度**：同步各 GPU 的梯度，每 GPU 得到全局平均梯度 $\bar{g} = \frac{1}{P}\sum g_i$
3. 每个 GPU 用 $\bar{g}$ 更新本地模型参数
4. 循环（每步后所有副本保持参数一致）

**PyTorch DistributedDataParallel（DDP）** 的优化：

- **Bucketed AllReduce**：参数按桶分组，每个桶在其所有梯度计算完毕后立即发起 AllReduce（与后续层反向传播重叠）
- **Gradient Compression**（可选）：AllReduce 前压缩梯度

**限制**：要求**单个 GPU 能容纳完整模型**（权重 + 梯度 + 优化器状态）。175B 模型的 Adam FP32 训练需要约 2.8TB，超过任何单 GPU。

### 12.2 Pipeline Parallelism（流水线并行）

**基本思路**：将模型按层切分，每 GPU 负责一段层（Stage），激活值在 Stage 间传递：

```
GPU 0: Layer 0–7   (Stage 0)
GPU 1: Layer 8–15  (Stage 1)
GPU 2: Layer 16–23 (Stage 2)
GPU 3: Layer 24–31 (Stage 3)
```

**朴素流水（Naive Pipeline）的"Bubble"问题：**

```
时间 →
GPU0: [F1][F2][F3][F4]                        [B4][B3][B2][B1]
GPU1:    [F1][F2][F3][F4]                  [B4][B3][B2][B1]
GPU2:       [F1][F2][F3][F4]           [B4][B3][B2][B1]
GPU3:          [F1][F2][F3][F4][B4][B3][B2][B1]
     ←───── Bubble（GPU 空闲）───────→
```

只有 GPU3 全程忙碌，其他 GPU 大量空转，硬件利用率极低。

**GPipe（Google，NeurIPS 2019）** 的解决方案：将 Batch 分成 $M$ 个 **Micro-batch**，流水处理：

```
（4 GPU，4 Micro-batch）
时间 →
GPU0: [F1][F2][F3][F4]                        [B4][B3][B2][B1]
GPU1:    [F1][F2][F3][F4]                  [B4][B3][B2][B1]
GPU2:       [F1][F2][F3][F4]           [B4][B3][B2][B1]
GPU3:          [F1][F2][F3][F4]    [B4][B3][B2][B1]

Bubble 占总时间的比例 ≈ (P-1)/(M+P-1)
当 M >> P 时，Bubble 比例 → 0
```

**代价**：需要更多内存来缓存 $M$ 个 Micro-batch 的中间激活（激活内存 ≈ $M$ 倍）。

**1F1B（One Forward One Backward）调度**（PipeDream 等采用）：进一步减少 Bubble 并控制内存。

### 12.3 Tensor Parallelism（张量并行）

**动机**：即使采用 Pipeline 并行，单个 Transformer 层的权重矩阵（如 8192×8192）可能仍太大（FP16 = 128MB/层），且层间激活传递使某些 GPU 频繁空转。Tensor Parallelism 将单个层的矩阵乘法拆到多 GPU 上并发计算。

**Megatron-LM（NVIDIA，2019）** 的方案（以 MLP 层为例）：

Transformer MLP 层：$Y = \text{GeLU}(XA)B$，其中 $A \in \mathbb{R}^{d \times 4d}$，$B \in \mathbb{R}^{4d \times d}$

**按列切分 A（Column Parallel）：**

$$A = [A_0 | A_1]，\quad XA = [XA_0 | XA_1]$$

GPU 0 计算 $XA_0$（维度 $\text{batch} \times 2d$），GPU 1 计算 $XA_1$（维度 $\text{batch} \times 2d$），**不需要通信**（输出直接局部存储）。

**按行切分 B（Row Parallel）：**

$$B = \begin{bmatrix}B_0\B_1\end{bmatrix}，\quad Y = XA_0 B_0 + XA_1 B_1$$

GPU 0 计算 $\text{GeLU}(XA_0) B_0$，GPU 1 计算 $\text{GeLU}(XA_1) B_1$，最后 **AllReduce** 合并：$Y = \text{GPU0_out} + \text{GPU1_out}$。

这样整个 MLP 层只需要一次 AllReduce（层末），而不是每次矩阵乘都通信。对于 Attention 层，QKV 矩阵也可以类似拆分。

### 12.4 ZeRO 与 FSDP

#### ZeRO（Zero Redundancy Optimizer）

**问题**：Data Parallel 下每个 GPU 有完整的模型状态（权重 + 梯度 + 优化器状态），大量冗余存储。

**ZeRO 的核心问题**：是否必须在每个 GPU 上始终保存完整副本？

**ZeRO 的三个阶段（内存节省随 Stage 升级）：**

|Stage|分片内容|内存节省（相对基础 DP）|
|---|---|---|
|**ZeRO-1**|优化器状态分片|~4×|
|**ZeRO-2**|优化器状态 + 梯度分片|~8×|
|**ZeRO-3**|优化器状态 + 梯度 + **参数**分片|~ $P$ ×（$P$ = GPU 数）|

**ZeRO-3 的工作原理（最彻底）：**

每个 GPU 只存 $1/P$ 的参数、梯度和优化器状态。

```
前向传播计算 Layer k 时：
  AllGather Layer k 的参数（所有 GPU 临时持有完整参数）
  → 计算 Layer k 的前向
  → 丢弃 Layer k 参数（释放内存）

反向传播计算 Layer k 时：
  AllGather Layer k 的参数（重新收集）
  → 计算 Layer k 的梯度
  → ReduceScatter 梯度（每 GPU 只保留自己负责的 1/P 梯度）
  → 丢弃 Layer k 参数
  → 用局部梯度更新局部参数
```

**代价**：ZeRO-3 的通信量约是 ZeRO-1/2 的 1.5×（因为前向传播也需要 AllGather），但换来的是接近线性的内存节省。

#### PyTorch FSDP（Fully Sharded Data Parallel）

FSDP 是 ZeRO-3 在 PyTorch 中的官方实现，名字来自"Fully Sharded"——参数、梯度、优化器状态全部分片（Shard）。

**FSDP 的 Unit 是 FlatParameter**：将若干层的参数 flatten 成一维张量，以此作为 AllGather/ReduceScatter 的基本单元，允许细粒度控制 Shard 边界。

**FSDP vs DDP 的对比：**

|特性|DDP|FSDP|
|---|---|---|
|内存使用|每 GPU 完整模型状态|每 GPU 1/P 模型状态|
|通信量|仅梯度 AllReduce|参数 AllGather × 2 + 梯度 ReduceScatter（约 1.5× DDP 通信）|
|最大可训练模型|~GPU 内存 / 16 bytes/param|理论上无限（但受通信带宽限制）|
|适用场景|模型能放进单 GPU|模型太大，需多 GPU 才能装下|

**CPU Offload**（FSDP/ZeRO 的可选扩展）：不活跃的参数和优化器状态卸载到 CPU RAM（~1TB）甚至 NVMe SSD（~TB），需要时再加载。代价是 CPU↔GPU 传输延迟，但允许训练比 GPU 内存大得多的模型。

### 12.5 3D Parallelism 与通用构建块

#### 3D Parallelism

在单个训练作业中**同时使用 Data Parallel + Pipeline Parallel + Tensor Parallel**：

```
假设 64 GPU，按 8 TP × 4 PP × 2 DP 组织：
  - 8 GPU 构成一个 Tensor Parallel 组（处理同一层的不同部分）
  - 4 个这样的 TP 组串联构成 Pipeline Parallel（处理不同层段）
  - 2 个这样的 PP 组并联构成 Data Parallel（处理不同 Batch 部分）
```

**通信层次：**

- TP 组内：AllReduce（延迟敏感，需要高带宽 NVLink）
- PP 组间：P2P（激活值传递，中等带宽）
- DP 组间：AllReduce（延迟不敏感，可接受 InfiniBand 延迟）

**设计原则**：将延迟敏感、通信频繁的 TP 放在高带宽 NVLink 连接的 GPU 内；PP 和 DP 可以跨节点（InfiniBand）。

#### 并行化的通用构建块

Shoeybi 等人指出所有 ML 并行策略都可以分解为以下三类操作（"The Next 700 ML Parallelization Schemes"）：

|操作|含义|例子|
|---|---|---|
|**Partitioning（分片/分割）**|将数据/算子切分到不同设备|ZeRO 参数分片，Pipeline 层分割|
|**Replication（复制）**|在不同设备上保留完整副本|DDP 的模型副本，参数服务器|
|**Loading/Unloading（装载/卸载）**|根据 Liveness 动态装载所需数据|FSDP 的 AllGather/ReduceScatter|

这三类操作作用于：

- **Operators / Layers**（算子层面的分割）
- **Data**（参数、梯度、优化器状态、激活值的管理）

**引入通信的时机**：

- 合并分片（Gather/AllGather/AllReduce）
- 设备间传递数据（P2P Send/Recv）

**允许的近似**：通信约束可以被适当放松——允许使用"过时的"数据（Stale Synchrony），如 Hogwild!、DistriFusion、某些 Pipeline 的 Gradient Staleness。

#### 性能诊断要点

排查分布式训练效率时，需要关注：

1. **通信量（Communication Volume）**：通信是否将 Compute-Bound 变成了 I/O Bound？
2. **计算强度（Computation Intensity）**：极端分片下每 GPU 的矩阵太小 → Tensor Core 利用率低（算术强度低于 Ridge Point）
3. **时间线分析**（使用 `nsight systems` 或 `torch.profiler`）：
    - GPU 空闲（Idle）时间（应最小化）
    - 过度同步（Barrier 等待）
    - 通信与计算是否有重叠

### 12.6 容错（Fault Tolerance）

#### 大规模训练中的故障概率

随着 GPU 数量增加，单个组件的可靠性决定了整个训练任务的 MTTF（平均故障间隔时间）：

$$MTTF_{cluster} \approx \frac{MTTF_{single}}{N}$$

以 10,000 个 GPU 为例，若每个 GPU 的 MTTF 为 100 天：$MTTF_{cluster} \approx 100/10000 = 0.01$ 天 = 14 分钟。即每 14 分钟就有约 1 次故障。长达数周的大模型训练必须有完善的容错机制。

**故障类型（Kokolis et al. 的分类）：**

- **硬件故障**：GPU 内存错误（ECC 可纠正/不可纠正），NVLink/InfiniBand 故障，电源故障
- **软件故障**：CUDA kernel 崩溃，Python OOM，死锁
- **基础设施故障**：数据中心断电，网络分区

#### 容错机制

**Checkpoint（检查点）**：定期（如每 30 分钟）将模型参数、优化器状态、数据加载器状态保存到分布式文件系统。发生故障后从最近的 Checkpoint 恢复。

**Goodput 最大化**：在故障概率与 Checkpoint 频率之间找平衡：

- Checkpoint 太频繁：I/O 开销大（每次保存 350GB 模型）
- Checkpoint 太稀疏：故障后丢失大量训练进度

**Singularity（Microsoft）** 的弹性训练（Elastic Training）：支持训练任务在运行中**动态增减 GPU**，无需重启：

- 新 GPU 加入：从 Checkpoint 恢复，加入 AllReduce 环
- GPU 故障退出：剩余 GPU 重新组织，继续从最近 Checkpoint 训练

---

## Module 13：Fine-Tuning — 参数高效微调

### 13.1 微调的动机与成本

**历史背景**：直到约 2018 年（GPT-1 之前），业界认为不同的 NLP 任务（翻译、摘要、问答）需要**不同的模型结构和从头训练**。

**GPT-1 的突破（Radford et al., 2018）**：将训练分为两阶段——

- **Pre-training（预训练）**：在大量无标注文本上训练通用语言模型（语言建模目标），一次完成
- **Fine-tuning（微调）**：以预训练权重为起点，在任务特定的有标注数据上继续训练，每个下游任务做一次

**成本差距**：

|阶段|GPT-3 规模的成本估计|
|---|---|
|预训练|~\$4–12M（一次性）|
|全量微调|~\$600（以 Alpaca 7B 为参考）|
|LoRA 微调|~\$50（以 Alpaca 7B-LoRA 为参考），单 GPU 数小时|

**全量微调的成本问题**：每个任务产生与原始模型等大的 Checkpoint（GPT-3 350GB）；多任务部署需要多份权重副本。**参数高效微调（PEFT, Parameter-Efficient Fine-Tuning）** 旨在仅训练极少量额外参数，同时达到与全量微调相当的任务性能。

### 13.2 Adapter（适配器）

**来源**：Houlsby et al., "Parameter-Efficient Transfer Learning for NLP"，NeurIPS 2019

**核心思路**：在 Transformer 的每层内插入**小型 Adapter 模块** $\phi_{v}$，原始权重 $w$ 冻结，只训练 $v$：

$$\text{原始层}: \psi_w(x)$$

$$\text{含 Adapter 的层}: \phi_{v}(\psi_w(x)) \approx \psi_w(x) \text{（初始时 } \phi \text{ 是恒等映射）}$$

**Adapter 的结构**（典型设计）：

```
输入 x
  ↓
Linear(d → r)     ← Down-projection（r << d，如 r=64，d=1024）
  ↓
Non-linearity（ReLU/GeLU）
  ↓
Linear(r → d)     ← Up-projection
  ↓
+x（残差连接）
  ↓
输出
```

**参数量分析**：约 $2rd$（远小于 Transformer 层的 $d^2$ 参数）。

**成本与代价：**

- 仅训练 Adapter 参数（约比 BERT 多 30% 参数，但只有这 30% 需要梯度计算）
- 推理时 Adapter 层必须执行，增加推理延迟
- 不同任务使用不同的 $v$，但 $w$ 共享——多任务推理只需加载不同的 Adapter，权重共享

### 13.3 Prefix Tuning（前缀调优）

**来源**：Li & Liang, "Prefix-Tuning: Optimizing Continuous Prompts for Generation"，ACL 2021

**核心思路**：不修改 Transformer 内部结构，而是在每层的 Key/Value 序列前**拼接可训练的"前缀"参数** $P_\theta$：

```
原始 Attention 输入：[x1, x2, ..., xn]
添加前缀后：         [p1, p2, ..., pk, x1, x2, ..., xn]
                      ↑ 可训练前缀（共 k 个 token 的 embedding）
```

**训练时**：前缀 $P_\theta$ 由一个 MLP 生成（避免直接优化不稳定），仅优化 MLP 参数。

**推理时**：将 MLP 的输出（固定的 $P_\theta$）预计算出来，拼接在输入序列前。效果上等同于每个 token 都"看到"了来自前缀的信息（Soft Prompt）。

**成本：** 约使用 0.1% 的任务特定参数（以 GPT-2 为参考），推理时只增加 $k$ 个 token 的 Attention 计算开销。

**Prompt Engineering vs Prefix Tuning**：Prompt Engineering（手写 In-context Prompt）是离散的（token 必须是词汇表中的词）；Prefix Tuning 的前缀是连续的（任意实数向量），通过梯度下降优化，表达能力更强。

### 13.4 LoRA（低秩适配）

**来源**：Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models"，ICLR 2022

**核心洞察**：预训练模型权重的更新矩阵 $\Delta W$ 往往是**低秩的**——即更新信息可以用两个小矩阵的乘积近似表示：

$$h = W_0 x + \Delta W x = W_0 x + BAx$$

其中：

- $W_0 \in \mathbb{R}^{d \times k}$：**冻结**的原始权重
- $A \in \mathbb{R}^{r \times k}$：**可训练**，随机初始化
- $B \in \mathbb{R}^{d \times r}$：**可训练**，初始为全零（保证初始时 $BA = 0$，不改变预训练行为）
- $r \ll \min(d, k)$：低秩维度（如 $r = 8$ 或 $r = 16$，而 $d = k = 4096$）

```
原始权重（冻结）：
x ──────────────────→ [W₀] ──→
                              + ──→ h
LoRA 旁路（可训练）：      ↗
x ──→ [A] ──→ [B] ──→
      (r×k)  (d×r)
```

**训练过程：** 只需对 $A$ 和 $B$ 计算梯度，$W_0$ 不需要梯度（大量节省内存和计算）。

**参数量分析：**

||GPT-3（175B，FP16）|使用 LoRA（r=4）|
|---|---|---|
|权重大小|350 GB|350 GB（冻结，不占训练显存）|
|可训练参数|175B（全量微调）|~35M（< 0.02%，约 70 MB）|
|训练所需梯度|全量 350 GB|约 70 MB|

**推理时的无缝合并：**

$$h = (W_0 + BA)x$$

可以在部署前**将 $W_0$ 和 $BA$ 合并**为单一权重矩阵 $W' = W_0 + BA$，推理时完全等同于原始模型，没有任何额外延迟！

这是 LoRA 相对 Adapter 的重大优势：Adapter 推理必须经过额外的 Adapter 层，而 LoRA 在推理时可以完全合并（或保持分离以支持多任务快速切换）。

**LoRA 的应用范围：**

- 通常应用于 Attention 层的 $Q, K, V, O$ 投影矩阵
- 也可以应用于 FFN 层
- 低秩维度 $r$ 通常是 1–64，越大越接近全量微调但参数量也越多

### 13.5 PEFT 方法对比

|方法|可训练参数量|推理开销|多任务切换|适用场景|
|---|---|---|---|---|
|**全量微调**|100%（$P$ 个参数）|无额外|每任务独立模型|有大量计算资源，追求最高性能|
|**Adapter**|~30%（BERT base）|有（额外层）|仅换 Adapter 权重|推理延迟不敏感|
|**Prefix Tuning**|~0.1%|很小（额外 $k$ token）|仅换前缀|生成任务，任务数多|
|**LoRA**|<0.1%|**无**（可合并）|合并前可换 $A,B$；合并后无法切换|几乎所有场景，是目前实践主流|
|**ReFT**|视配置|有（额外操作）|是|新方向，研究性|

**QLoRA**（量化 + LoRA）：在 4-bit 量化的冻结权重基础上训练 LoRA 模块，进一步减少内存需求。例如，65B 参数模型可以在 1 张 48GB A40 上微调。

---

## Module 14：Mixture of Experts

### 14.1 MoE 的动机与基本公式

#### Scaling Law 的两难困境

**Scaling Law** 的核心结论（Kaplan et al., 2020）：模型性能随参数量、数据量、计算量的增加而改善，且有明确的幂律关系。直接推论：**更大的模型 = 更好的性能**。

**但**：训练一个 $n$ 倍大的稠密（Dense）模型需要约 $n$ 倍的计算量（每个 Token 都经过所有参数）。从 GPT-3（175B）到 GPT-4 级别的模型，计算成本呈爆炸式增长。

**Mixture of Experts 的出发点**：能否构建一个**参数量很大但每次计算只用其中一小部分**的模型？

#### MoE 的数学定义

$$y = \sum_{i=1}^{n} G(x)_i \cdot E_i(x)$$

其中：

- $E_i(x)$：第 $i$ 个 **Expert**（一个小型 FFN 网络）
- $G(x)_i$：**Gate 网络**（路由网络）对第 $i$ 个 Expert 的权重
- $n$：Expert 总数（通常 8–64 个，极端情况如 DeepSeekMoE 有 256 个）

**关键**：若 $G(x)_i = 0$，则 Expert $E_i$ 完全不需要计算！通过使 $G$ 产生**稀疏输出**（Top-K 选择，$K \ll n$），每个 Token 只激活 $K$ 个 Expert，实现**条件计算（Conditional Computation）**。

**Sparsely-Gated MoE（Shazeer et al., ICLR 2017）** 的门控函数：

$$G(x) = \text{Softmax}(\text{Top-K}(x \cdot W_g + \text{noise}))$$

只有 Top-K 个值非零，其他 Expert 不计算。$\text{noise}$ 是训练时加的噪声，帮助探索（防止少数 Expert 主导）。

**稀疏 vs 稠密的性能对比（以 1B 激活参数为例）：**

|模型类型|总参数|每 Token 激活参数|计算量/Token|模型质量|
|---|---|---|---|---|
|Dense 1B|1B|1B|1×|基准|
|MoE 8B（Top-2，8 Expert）|8B|1B（约）|~1×|明显好于 Dense 1B，接近 Dense 8B|
|Dense 8B|8B|8B|8×|好|

MoE 的优势：以 Dense 1B 的计算量获得接近 Dense 8B 的模型质量。

### 14.2 训练 MoE 的系统挑战

#### Shrinking Batch Problem（批大小缩减问题）

**问题**：假设 batch size = $b$，有 $n$ 个 Expert，每个 Token 选 Top-$k$ 个 Expert。则平均每个 Expert 接收约 $\frac{b \cdot k}{n}$ 个 Token。

当 $n$ 很大（如 64 个 Expert），每个 Expert 的有效 batch size 仅为 $b/n \cdot k$ ——极小！

**为什么小 batch 有问题**：回忆 Roofline 模型——矩阵乘法 $Y = XW$，当 $X$ 的 batch 维度（$b$）很小时，算术强度 $I = \frac{2 \cdot b \cdot d^2}{2 \cdot b \cdot d + 2 \cdot d^2} \approx b$（当 $b \ll d$ 时）。小 $b$ → 小 $I$ → Memory-Bound → GPU 计算单元闲置。

**解决方案（Expert Parallelism + 批次路由）：**

```
不用 Expert Parallelism（单设备）：
- 每个 Expert batch size = b/n，严重 Compute-Bound 以下
- 效率极低

使用 Expert Parallelism（d 个设备，每设备一个 Expert 副本）：
- 将多个 Data Parallel 设备的 Batch 汇聚
- 每个 Expert 接收的 Token 数变为 (b × d × k) / n
- 需要 AlltoAll 将 Token 路由到对应设备
```

即通过增加设备数量来增加每个 Expert 的有效 Batch Size，维持合理的算术强度。

#### 网络带宽（Network Bandwidth）

MoE 的 Expert 分散在不同设备上，Token 需要通过网络发送到对应 Expert，Expert 的输出也需要发回——这是额外的通信开销，不存在于 Dense 模型中：

```
Token 路由流程：
GPU 0 的 Token → [Gate] → Expert_3（在 GPU 3）→ [网络传输] → 计算 → [网络传输] → GPU 0 的输出
```

通信量与 Expert 隐藏层维度（Hidden Dim）成正比，可以通过减小 Expert 中间层大小来减少通信。

#### Load Balancing（负载均衡）

**问题**：如果 Gate 网络"偏爱"某几个 Expert，会导致：

- 热门 Expert：Token 过多，排队等待（高延迟）
- 冷门 Expert：Token 很少，GPU 空闲（低利用率）
- 训练时：热门 Expert 接收更多训练信号 → 更强 → 更多 Token 被路由到它（**马太效应**）

**解决方案**：在训练损失中加入 **Auxiliary Load Balancing Loss**，惩罚 Expert 负载不均：

$$L_{aux} = \alpha \cdot n \cdot \sum_{i=1}^{n} f_i \cdot P_i$$

其中 $f_i$ 是 Expert $i$ 实际接收的 Token 分数，$P_i$ 是 Gate 对 Expert $i$ 的平均概率。鼓励 $f_i$ 均匀分布（每个 Expert 接收 $1/n$ 的 Token）。

### 14.3 GShard 与 DeepSeekMoE

#### GShard（Google，2020）

**主要贡献**：将 MoE 应用于 Transformer，实现 6000 亿参数的模型（每 Token 激活约 600 亿参数），同时提出了 MoE 的系统化实现框架：

- **Compiler-based implementation**：用户通过注解指定哪些层是 MoE，编译器自动生成分片和通信代码
- **O(1) 内存（相对 Expert 数）**：每设备只存 $1/n$ 的 Expert
- **随机路由（Random Routing）**：发送超过容量的 Token 随机路由到其他 Expert（而非截断），略微降低精度但提高负载均衡
- **Capacity Factor**：限制每个 Expert 在一个 Batch 内最多处理 $C = \text{capacity\_factor} \times \frac{b \cdot k}{n}$ 个 Token，超出的 Token 直接残差连接跳过 Expert（Overflow 机制）

#### DeepSeekMoE（DeepSeek，2024）

DeepSeekMoE 针对"普通 MoE 中 Expert 学到的知识过于通用"这一问题提出两个主要改进：

**① 细粒度 Expert 分割（Fine-grained Expert Segmentation）：**

假设：如果 Expert 数量少，每个 Expert 为了覆盖足够多的 Token，被迫学习宽泛的通用知识，造成 Expert 间的知识冗余。

解决：将每个 Expert 的 FFN 中间维度缩小 $m$ 倍，同时增加 $m$ 倍 Expert 数量，并激活 $m$ 倍的 Expert：

$$\text{原始}: n \text{ 个 Expert，每个 FFN 维度 }d_{ffn}，\text{激活 }K \text{ 个}$$

$$\text{改进}: mn \text{ 个 Expert，每个 FFN 维度 }\frac{d_{ffn}}{m}，\text{激活 }mK \text{ 个}$$

计算量相同，但 Expert 更专一（每个 Expert 接收更少、更专精的 Token）。

**② 共享 Expert（Shared Experts）：**

假设：每个普通 Expert 都会为了处理通用的共性知识（如基本语法）消耗容量，造成 Expert 学习效率低。

解决：设置 $K_s$ 个**常驻 Expert**（Shared Experts），对所有 Token 都激活，专门处理通用知识。动态 Expert 的激活数量减少 $K_s$ 个：

```
每个 Token 的路由：
  Shared Experts: [E_s1, E_s2, ..., E_{Ks}]  ← 所有 Token 都激活这些
+
  Dynamic Experts: Top-(K - K_s) 个           ← 路由到最相关的动态 Expert
```

类似于 **Residual MoE（DeepSpeed-MoE 也有类似机制）**：共享 Expert 学基础，动态 Expert 学专精。

**DeepSeek-V3 的实现规模**：671B 总参数，每 Token 激活约 37B 参数；256 个路由 Expert + 1 个共享 Expert（per MoE 层）。

### 14.4 MoE 推理优化

#### DeepSpeed-MoE 推理（Microsoft，2022）

**推理瓶颈分析**：

MoE 推理与 Dense 推理的关键差异在于——Dense 模型每层的权重对所有 Token 相同（可以批次化）；MoE 模型每个 Token 路由到不同 Expert（分散化）。

小 Batch 推理时（$b = 1$，典型 LLM 推理场景）：

$$I_{dense} = \frac{2d}{2d + 2} \approx 1 \text{ FLOP/Byte（极端 Memory-Bound）}$$

MoE 每个 Expert 接收 $b/n$ 个 Token，算术强度更低——即使比 Dense 节省了总计算量，每个 Expert 的效率极差。

**DeepSpeed-MoE 的三个核心优化：**

**① 专家并行与路由共定位（Expert Parallelism + Critical Path Routing）：**

- 将计算路径相同的 Token（即路由到相同 Expert 序列的 Token）放在同一设备上处理
- 使用 Expert Parallelism 增大每个 Expert 的有效 Batch Size（见 14.2）
- Tensor Slicing（张量切片）在 Non-Expert 层（如 Attention）内并行

**② 分层 AlltoAll（Hierarchical AlltoAll）：**

传统 AlltoAll：$P$ 个设备两两通信，每对都发送数据，总通信跳数多。

```
传统（全对全）：
设备0 ↔ 设备1 ↔ 设备2 ↔ ... ↔ 设备P
（每对直连，高延迟）

分层（先机内 AlltoAll，再机间 AlltoAll）：
机器A内：[GPU0, GPU1, ..., GPU8] 做 AlltoAll（NVLink，低延迟）
机器B内：[GPU0, GPU1, ..., GPU8] 做 AlltoAll（NVLink，低延迟）
机器A↔B：汇总后的数据通过 InfiniBand 传输（只需一次，减少跨机通信）
```

**分析**：小 Batch 下，通信是**延迟约束（Latency-bound）**而非**带宽约束（Bandwidth-bound）**（总数据量小，但每个通信操作的启动开销固定）。分层 AlltoAll 减少了高延迟的跨机器通信次数，用 NVLink 的低延迟机内通信替代。

**③ MoE 专用 Kernel（Fused MoE Kernels）：**

Gate 计算包含多个步骤：Top-K 选取 → Softmax → Cumulative Sum（分配 Token 偏移）。这些步骤在朴素实现中是多个独立 kernel，大量额外的 Global Memory 读写。融合后一次 kernel 完成所有操作：

- **Blelloch Scan（Prefix Sum）**：用并行前缀扫描算法（$O(\log n)$ 时间）代替串行 Cumulative Sum

---

## Part II 总结

### 各模块核心概念速查

|模块|核心概念|关键公式/原理|
|---|---|---|
|**Module 6**|计算图、算子类型、图优化|Fusion = 减少 HBM 读写；DAG 调度|
|**Module 7**|Loop 变换（Tiling, Unrolling, Fusion）|Tiling 的核心：$\text{Tile Size} \approx \text{Cache Capacity}^{1/2}$ |
|**Module 8**|Roofline 模型、延迟隐藏| $P = \min(T_{peak}, \beta \times I)$；$n = R \times t$ |
|**Module 9**|内存管理、量化、Checkpointing|训练内存 = $16P$ bytes/param（Adam FP32）|
|**Module 10**|Collective Comm、Ring AllReduce、NCCL|Ring AllReduce 通信量 $\approx 2|
|**Module 11**|自动微分（Reverse AD）、SGD|Reverse AD：1 次 backward = 所有参数梯度|
|**Module 12**|DP/PP/TP/FSDP/ZeRO|ZeRO-3 内存 = $(1/P)$ 倍 DP；FSDP = ZeRO-3 的 PyTorch 实现|
|**Module 13**|LoRA、Adapter、Prefix Tuning|LoRA 参数量 $= 2rd$，推理无额外开销（可合并）|
|**Module 14**|MoE 稀疏激活、负载均衡、专用推理优化|MoE：大参数量 + 小激活量 = 高效扩展|

### Part II → Part III 的衔接

Part II 建立了完整的 ML 系统效率框架，Part III（Advanced Topics）在此基础上深入具体前沿话题：

- **Roofline 模型** → 理解 FlashAttention、vLLM、量化的优化动机
- **通信与并行化** → 理解 DistriFusion（扩散模型分布式推理）的通信放松机制
- **LoRA 与量化** → 理解 QLoRA、GPTQ、Marlin INT4 GEMM kernel
- **Loop Tiling + Triton** → 进阶：自己编写高性能 ML kernel
- **MoE 系统** → 理解 DeepSeek-V3、Mixtral 的工程实现

# Part III：高级专题（Advanced Topics）

> **说明**：Part III 是论文研讨课（Paper Discussion）模式，不同于 Part I/II 的结构化讲授。本文档按主题组织，对每篇核心论文给出系统性笔记，并辅以必要的背景理论。

## 附录：可扩展性理论（Scalability Compendium）

> 对应 Fall 2024 Compendium 讲次，是理解 Scaling 话题的理论基础。

### A. 系统优化的通用策略框架

所有使系统变快的手段，最终归结为对性能模型 $T = \frac{W \times t}{P}$ 三个变量的操纵：

|策略方向|作用|具体手段|
|---|---|---|
|**减少工作量 $W$ **|少做事|更好算法（$O(n)$ 替代 $O(n^2)$）、避免冗余计算、算子融合|
|**降低单位成本 $t$ **|每件事做得快|向量化、Cache 优化（Tiling）、低精度（FP16 / INT8）、数据压缩|
|**增加并行度 $P$ **|同时做多件事|多核 / 多 GPU / 多机、流水线、SIMD|

**分类更细的工作类型：**

- **Compute work**：算术运算、逻辑运算
- **Memory work**：Load / Store 指令
- **Communication work**：设备间 / 机器间的数据传输

减少各类工作的手段各不相同：

```
减少 Memory work：
  ├── 用更小的数据类型（FP32→FP16：减少 2× 内存流量）
  ├── 量化（FP32→INT8：减少 4× 内存流量）
  ├── 压缩（无损 / 有损）
  └── 提升局部性（Tiling：使更多数据命中 Cache，而非访问 HBM）

减少 Communication work：
  ├── 避免通信（让计算和数据在同一设备上）
  ├── 每 k 步发送一次（梯度累积）
  └── 使用过期数据（Stale Synchrony）
```

### B. Amdahl's Law（强可扩展性）

$$\text{Speedup}(P) = \frac{T_1}{T_P} = \frac{1}{\frac{\alpha}{P} + (1 - \alpha)}$$

其中 $\alpha$ 是程序中**可并行化**的比例，$(1-\alpha)$ 是**串行瓶颈**的比例，$P$ 是处理器数量。

**极限分析**（$P \to \infty$）：

$$\text{Speedup}_{\max} = \frac{1}{1 - \alpha}$$

|可并行比例 $\alpha$ |最大加速比|
|---|---|
|50%|2×|
|80%|5×|
|90%|10×|
|95%|20×|
|99%|100×|
|99.9%|1000×|

**核心教训**：哪怕 1% 的串行代码，最终会成为扩展到数千 GPU 时的决定性瓶颈。

**串行部分从哪里来？**

- **I/O 串行化**：读取训练数据文件必须在 Worker 线程分发前完成
- **同步点（Barriers）**：AllReduce 需要等所有 GPU 完成反向传播
- **资源竞争**：所有请求流向同一参数服务器
- **单点瓶颈**：Pipeline Parallel 中最慢的那个 Stage

**对 ML 训练的含义**：数千 GPU 规模的训练中，每一个 Barrier、每一次锁、每一个串行 Checkpoint 写入都可能成为决定性瓶颈。这解释了为什么 FSDP、异步梯度更新、Overlap 等技术如此重要。

### C. Gustafson's Law（弱可扩展性）

Amdahl's Law 假设问题规模不变、增加处理器数量，关注**固定规模的加速比**（Strong Scaling）。

Gustafson's Law 提出另一个视角——**随着资源增加，也增大问题规模**（Weak Scaling）：

$$\text{Speedup}(N) = s + Np$$

其中 $s$ 是串行部分时间，$p$ 是并行部分时间（归一化为 $s + p = 1$），$N$ 为处理器数。

**直觉**：更大的机器能在**同样的时间内**解决**更大的问题**。

**对 ML 的应用**：Scaling Laws 的核心逻辑其实是 Gustafson 视角——用更多 GPU 不是为了更快地训练同一个模型（Strong Scaling），而是在同样的训练时间内训练更大的模型（Weak Scaling）。这是"训练更大模型 = 更好性能"这一信条的系统性基础。

---

## Topic A：高性能 ML 代码生成

### A.1 Triton：Tiled 神经网络计算的编译器

> **论文**：Tillet, Kung, Cox, "Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations," MAPL 2019。PhD 扩展版：Tillet，"Blocked Algorithms for Neural Networks," Harvard, 2020

#### 核心问题

手写高性能 CUDA Kernel 极其困难：程序员需要同时管理：

- Shared Memory 的加载和存储
- Warp 内的 Coalesced Memory Access
- 寄存器的分配与 Spilling
- 向量化与 Bank Conflict 规避
- Tensor Core 的调用格式（wmma / mma 指令）

而且这些优化**高度硬件特定**——对 A100 优化的代码在 H100 上可能不是最优的，更别说 AMD GPU。

#### Triton 的解决方案：以 Tile 为核心抽象

Triton 在 CUDA 线程模型和高层 Python 之间引入了一个**中间抽象层**：

```
程序员的视角（Triton 代码）：
  - 每个 Program（对应一个 Thread Block）处理一个 Output Tile
  - 用 `tl.load`、`tl.store`、`tl.dot` 等操作 Tile
  - Tile 大小作为编译时常量（constexpr）

编译器的工作：
  - 分析 Tile 的访问模式，自动生成 Shared Memory 加载
  - 自动插入 __syncthreads()（线程同步）
  - 自动向量化 Tile 内的操作
  - 自动规避 Bank Conflict（选择访问步长）
  - 自动为 Tensor Core 生成 wmma/mma 指令
```

**矩阵乘法的 Triton 实现示例（官方文档版本，已简化）：**

```python
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    A, B, C,              # 三个矩阵的指针
    M, N, K,              # 矩阵维度
    stride_am, stride_ak, # A 的步长（行和列方向）
    stride_bk, stride_bn, # B 的步长
    stride_cm, stride_cn, # C 的步长
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # 每个 Program 负责计算 C 中一个 BLOCK_M × BLOCK_N 的输出 Tile
    pid_m = tl.program_id(0)  # 行方向的 Program 编号
    pid_n = tl.program_id(1)  # 列方向的 Program 编号

    # 计算输出 Tile 的起始偏移
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # 初始化累加器（在寄存器中，而非显存）
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K 维度方向的循环（对 A 和 B 的对应 Tile 做乘加）
    for k in range(0, K, BLOCK_K):
        # 加载 A 的 Tile：[BLOCK_M, BLOCK_K]
        a = tl.load(A + offs_m[:, None] * stride_am + (k + offs_k)[None, :] * stride_ak,
                    mask=(offs_m[:, None] < M) & ((k + offs_k)[None, :] < K))
        # 加载 B 的 Tile：[BLOCK_K, BLOCK_N]
        b = tl.load(B + (k + offs_k)[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                    mask=((k + offs_k)[:, None] < K) & (offs_n[None, :] < N))
        # Tile 内矩阵乘（编译器映射到 Tensor Core mma 指令）
        acc += tl.dot(a, b)

    # 写回输出
    tl.store(C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
             acc.to(tl.float16),
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

# 调用：将 M×N 的输出划分为 (M/BLOCK_M) × (N/BLOCK_N) 个 Program
grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
matmul_kernel[grid](A, B, C, M, N, K, ...)
```

**Triton 编译器做的事（程序员看不到）：**

1. 分析 `offs_m`、`offs_k` 的结构，确定内存访问是 Coalesced 的（连续地址）
2. 将 `a`（`BLOCK_M × BLOCK_K`）和 `b`（`BLOCK_K × BLOCK_N`）的 `tl.load` 编译为：先将数据加载到 Shared Memory，再从 Shared Memory 传入 Tensor Core 输入寄存器
3. 自动插入 `__syncthreads()` 防止 Shared Memory 的读写竞争
4. 将 `tl.dot` 编译为最优的 `mma` 指令序列（根据 Tile 大小选择 mma 变体）
5. 根据 Tile 大小和 GPU 架构，自动管理寄存器分配

**Triton vs 手写 CUDA：**

|维度|手写 CUDA|Triton|
|---|---|---|
|代码量|数百行|数十行|
|性能（A100 FP16 MatMul）|~85–95% 峰值（最优实现）|~80–90% 峰值|
|可移植性|极差（针对特定 GPU）|好（支持 NVIDIA/AMD）|
|开发效率|低|高|

**Triton 的局限：**

- 当前只支持 GPU（不支持 CPU）
- 不支持所有 CUDA 功能（如动态共享内存分配）
- 对于极度优化的 FlashAttention 等 kernel，手写 CUDA 仍有优势（需要更细粒度控制）

**现实影响**：PyTorch 2 的 TorchInductor 后端默认生成 Triton Kernel；xFormers、Flash-Attention 2 的大量实现基于 Triton；Flash Decoding 等推理优化也依赖 Triton。Triton 正在成为 ML Kernel 开发的事实标准。

---

### A.2 PyTorch 2：动态字节码变换与图编译

> **论文**：Ansel et al., "PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation," ASPLOS 2024

#### 核心挑战

PyTorch 1.x 的 Eager 模式（每个算子立即执行）面临以下性能限制：

1. **Python 解释器开销**：每次 Python 函数调用都需要 CPython 的 bytecode 解释、引用计数、GIL 等
2. **无图级优化**：相邻算子不能融合，中间张量必须完整写回 HBM
3. **动态性障碍**：Python 的动态特性（条件分支、可变大小输入、Python 控制流）使静态图难以捕获

过去的解决方案（TorchScript、`torch.jit.trace`）要求程序员**改写代码**，用受限的静态子集替代完整 Python——用户接受度差。

#### PyTorch 2 的核心技术：TorchDynamo

**设计目标**：在**不改变用户代码**的前提下，自动将 PyTorch 程序捕获为可优化的计算图。

**关键洞察**：Python 解释器的字节码（bytecode）是可以被拦截和修改的。TorchDynamo 在 CPython 的 Frame Evaluation API 层面注入钩子，**动态地重写 Python bytecode**：

```python
# 用户代码（完全不需要修改）
@torch.compile       # 这一行装饰器触发 TorchDynamo
def my_forward(x, y):
    z = torch.matmul(x, y)       # Tensor op
    if z.sum() > 0:              # Python 控制流！
        return torch.relu(z)
    else:
        return z

# 背后发生的事：
# 1. TorchDynamo 捕获第一次调用时的 bytecode
# 2. 识别出 Tensor op 序列（matmul, sum, relu）
# 3. 在 if/else 处"断图"（Graph Break）——控制流依赖于 Tensor 值，无法静态确定
# 4. matmul + relu 被编译为 FX Graph → TorchInductor → Triton Kernel
# 5. Python 中的 if/else 保持动态执行，仅在 True 分支调用编译后的 kernel
```

**Guard 机制（动态重编译）：**

TorchDynamo 为每次编译的图添加**Guard**（守卫条件）：一组输入必须满足的约束（如：`x.shape == (32, 512)`，`x.dtype == torch.float16`）。

```
第一次调用（x.shape = (32, 512)）：
  → 编译 Graph_A，绑定 Guard: shape==(32,512), dtype==fp16

第二次调用（x.shape = (32, 512)）：
  → Guard 命中，直接运行 Graph_A（无编译开销）

第三次调用（x.shape = (64, 512)）：
  → Guard 失效，重新捕获并编译 Graph_B，绑定 Guard: shape==(64,512)
```

Guard 机制使 PyTorch 2 能处理**动态形状**（如变长序列），代价是首次运行（或形状改变时）有编译延迟（Warmup Cost）。

**后端编译器层次：**

```
用户 Python 代码
    ↓ TorchDynamo（bytecode 分析 + 图捕获）
FX Graph（PyTorch 中间表示，类似 DAG）
    ↓ 图变换（融合、形状分析）
    ↓ TorchInductor（默认后端）
Triton Kernel（GPU）/ C++ + OpenMP（CPU）
    ↓
硬件执行
```

**TorchInductor 的工作：**

1. 接收 FX Graph
2. 应用算子融合（Pointwise + Reduction 融合等）
3. 为每个融合后的算子生成 Triton Kernel（或 CPU 的 C++ + OpenMP 代码）
4. 自动进行 Loop 变换（Tiling、Unrolling）

**实测性能提升（来自论文）：**

在 163 个开源模型上测试（BERT、ResNet、GPT-2 等），torch.compile 平均加速 **43%**（与 Eager Mode 对比），部分模型（如 BERT 推理）加速可达 2× 以上。

**Graph Break（图断裂）问题：**

`print()` 调用、自定义 Python 类的特殊方法、复杂 Python 控制流（依赖 Tensor 值的 if-else）等会导致 Graph Break——TorchDynamo 在断裂点停止图捕获，退回 Eager 执行。每次 Graph Break 都意味着一段代码无法被 TorchInductor 优化。

实践建议：用 `torch._dynamo.explain(model, *inputs)` 检查哪些位置发生了 Graph Break，并通过重构代码减少 Break。

---

## Topic B：LLM 推理服务系统

### B.1 vLLM / PagedAttention：LLM 推理的内存管理

> **论文**：Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention," SOSP 2023

#### 背景：LLM 推理的独特挑战

LLM 推理（自回归解码）与训练完全不同，面临特有的系统挑战：

**KV Cache 的问题**：Transformer 每个 Attention 层需要保存所有**已生成 Token** 的 Key 和 Value（用于下一 Token 的 Attention 计算）：

```
生成第 t 个 Token：
  Attention(Q_t, [K_1,...,K_t], [V_1,...,V_t])
  ↑ 需要所有历史 K 和 V
```

KV Cache 的大小随序列长度**线性增长**：

$$|KV \text{ Cache}| = 2 \times n_{layers} \times n_{heads} \times d_{head} \times L \times \text{sizeof}(\text{dtype})$$

以 LLaMA-13B 为例（FP16，序列长度 2048）： $$2 \times 40 \times 40 \times 128 \times 2048 \times 2 \approx 1.6 \text{ GB}$$

**问题一：静态预分配浪费大量显存**

传统做法：为每个请求**预分配**最大可能序列长度的 KV Cache（因为不知道最终会生成多长）。实际上大多数请求远短于最大长度，导致大量 KV Cache 显存被浪费。

**问题二：显存碎片化**

不同请求占用不同大小的 KV Cache，释放短请求后留下的碎片无法被长请求利用：

```
显存状态（每格 = 一定大小显存）：
[请求A KV:███][请求B KV:██████][空闲:██][请求C KV:████]
                                         ↑ 碎片
新请求需要 5 格：空闲区域太小，尽管总空闲 ≥ 5 格！
```

**问题三：Prefix 共享**

多个请求可能共享相同的系统提示（System Prompt）或对话历史，但传统实现对每个请求单独分配 KV Cache，无法共享这部分内容。

#### PagedAttention：借鉴虚拟内存分页

**核心思想**：将 KV Cache 分成固定大小的**Page（页面）**（每 Page = K 个 Token 的 KV Cache，如 K=16），通过一张**Page Table** 管理物理 KV Cache 块到逻辑序列位置的映射——完全借鉴操作系统的虚拟内存机制。

```
逻辑 KV Cache（一个请求的视角）：
Token 0-15: Page 0 → 物理 Block 7
Token 16-31: Page 1 → 物理 Block 2
Token 32-47: Page 2 → 物理 Block 15
...

物理 KV Cache 显存：
Block 0  Block 1  Block 2  Block 3  ...  Block 15  ...
[空闲]   [空闲]   [请求B P1] [请求A P0] ... [请求B P2] ...
```

**PagedAttention 解决的三个问题：**

1. **消除内部碎片**：每次只分配所需的 Page，不预分配多余空间。最后一个 Page 可能有部分浪费，但平均浪费 < 0.5 Page / 请求（相比传统方案浪费数百 MB）
    
2. **消除外部碎片**：物理 Block 可以被任何请求的任何 Page 使用，类似于 OS 的物理内存页框管理
    
3. **Prefix Sharing（Copy-on-Write）**：共享相同系统提示的多个请求，可以将对应的物理 Block 设为只读共享，只有在需要写入时才触发 Copy-on-Write 创建私有副本——与 Linux 的 `fork()` 原理相同
    

**实现 PagedAttention 的技术挑战**：标准的 Attention CUDA Kernel 假设 KV Cache 是**连续的内存块**，PagedAttention 需要专用 Kernel 支持**非连续、分散的物理块**：

```cuda
// 标准 Attention（假设连续内存）：
attn_score = Q × K_cache[0:seqlen]     // 连续地址

// PagedAttention（分散块）：
for page_idx in page_table[request]:
    block_ptr = kv_block_pool[page_idx]  // 可能是任意物理地址
    attn_score += Q × block_ptr->K       // 需要间接访问
```

vLLM 提供了专用的 Paged Attention CUDA Kernel，支持上述分散访问。

**vLLM 的其他系统设计：**

- **Continuous Batching（连续批处理）**：不等所有请求同时完成，已完成的请求立即释放 Page，新请求立即填入，提高 GPU 利用率（对比传统的 Static Batching：等一个 Batch 内所有请求都完成才处理下一批）
- **Preemption（抢占）**：当显存不足时，挂起低优先级请求（将其 KV Cache Swap out 到 CPU RAM 或直接丢弃），优先服务高优先级请求

**实测效果**（来自论文，A100 40GB，LLaMA-13B）：

- 与 FasterTransformer 对比：吞吐提升 **2.2×**（单序列），**3.5×**（多序列）
- 显存浪费从传统方案的 ~60–80% 降至 **<4%**

---

### B.2 量化推理：QMoE 与主流量化方法

> **核心论文**：Frantar & Alistarh, "QMoE: Practical Sub-1-Bit Compression of Trillion-Parameter Models," MLSys 2024

#### 量化的系统性动机

以 LLaMA-70B 推理为例，假设 FP16 精度、batch size=1：

- 权重大小：70B × 2 bytes = 140 GB
- 单次前向传播：约 140 GB 权重需从 HBM 读取一次
- A100 HBM 带宽：2 TB/s
- **纯加载时间下界**：140 / 2000 = 0.07 秒 = 70 ms/token

这意味着仅权重加载就决定了 batch=1 时的最大推理速度约为 14 tokens/s。量化到 INT4 后权重大小变为 35 GB，理论上限提升到 57 tokens/s。这是 Weight-only Quantization 的核心价值。

#### 主流量化方法

**GPTQ（Post-Training Quantization via Layer-wise OBQ）：**

每层权重量化时，通过求解最小化量化误差的优化问题：

$$\min_{W_q} |XW - XW_q|_F^2$$

使用 Hessian 信息（$H = X^T X$）指导量化——更重要的权重（对 Hessian 特征值贡献大的）用更高精度。GPTQ 对大模型（6.7B+）效果好，但对小模型（<3B）有较大精度损失。

**AWQ（Activation-Aware Weight Quantization）：**

观察：少数（约 1%）权重对激活值分布影响极大（对应"显著"通道）。AWQ 识别这些显著权重，对其进行**缩放**（Scaling）而非量化，从而在 INT4 量化下保留精度：

$$W_{q} = \text{Quantize}(W \cdot s^{-1}) \cdot s$$

其中 $s$ 是每通道的缩放因子，通过分析激活值分布确定。AWQ 比 GPTQ 更适合小模型，且速度更快（不需要求解复杂优化问题）。

**Marlin（INT4 GEMM Kernel for Weight-only Quantization）：**

GPTQ/AWQ 量化后的 INT4 权重需要在 Tensor Core 上高效执行矩阵乘法。但 NVIDIA Tensor Core 不原生支持 INT4 × FP16 的混合精度 GEMM（INT4 权重需要先反量化到 FP16，再做 FP16 GEMM）。

**Marlin 的贡献**：为 A100/4090 等设计了专用的 INT4 × FP16 GEMM Kernel，核心思路：

1. **权重重排（Weight Reordering）**：将 INT4 权重预处理为最优的物理内存布局，以便在 Warp 内实现 Coalesced Load
2. **异步 IO 与计算**：在计算当前 Tile 的同时，异步加载下一 Tile 的权重（Prefetching）
3. **分组量化（Grouped Quantization）**：每 group_size=128 个权重共享一组缩放因子，平衡精度和开销

**Marlin 的 Batch-size 行为**（与你的 Profiler 项目直接相关）：

```
Batch=1 时：
  每次 GEMM 只需要读取对应的 1 行输出，输入激活向量很小
  → 算术强度 I ≈ 1 FLOP/Byte（极度 Memory-Bound）
  → Marlin 的主要作用：INT4 权重比 FP16 小 4×，HBM 带宽需求降低 4×
  → 吞吐 ≈ 原来的 4×（接近理论上限）

Batch=16 时：
  每次 GEMM 读取 16 行，激活矩阵变大
  → 算术强度 I ≈ 16 FLOP/Byte（仍 Memory-Bound，但更高）
  → Marlin 优势略降（带宽节省仍然显著）

Batch=128 以上：
  激活矩阵足够大，算术强度越过 Ridge Point
  → 进入 Compute-Bound 区间
  → 此时 INT4 GEMM 的 Tensor Core 计算速度决定性能（而非带宽）
  → Marlin 相对 FP16 GEMM 的优势取决于 INT4 Tensor Core 的峰值算力
```

这个 batch-size 分界点（大约在 batch=64-128，视 GPU 型号而定）就是 Marlin 性能曲线的"拐点"——也是你的 Quantization Tax Profiler 应该精确测量并可视化的核心现象。

#### QMoE：Trillion 参数模型的 Sub-1-Bit 压缩

**动机**：Switch Transformer（1.6T 参数）等 MoE 模型即使以 INT8 存储也需要约 1.6 TB，根本无法装进任何当前 GPU 集群。

**QMoE 的关键洞察**：

1. **MoE 的 Expert FFN 权重极其可压缩**：Transformer 中，Attention 层的权重（Q/K/V/O 投影）对精度损失敏感；但 MoE 的 Expert FFN 权重对压缩更鲁棒，可以用极低精度（< 1 bit/weight）量化
2. **结合 SparseGPT 与自定义编码**：用 SparseGPT（基于 Hessian 的权重剪枝）对 Expert 权重进行稀疏化，再对非零权重做极端量化（3 bits），整体等效 < 1 bit/weight

**结果**：Switch-c2048（1.6T 参数）压缩到约 160 GB（约 0.8 bits/weight），可以在 2 台 A100 节点（8 × 80GB GPU）上运行推理，精度损失在可接受范围内（困惑度 Perplexity 仅小幅上升）。

**QMoE 的系统贡献**：专用的压缩格式和 Decompression Kernel，在推理时动态解压，避免将完整权重展开到 HBM（因为完整展开后仍然放不下）。

---

### B.3 DistriFusion：扩散模型的分布式推理

> **论文**：Li et al., "DistriFusion: Distributed Parallel Inference for High-Resolution Diffusion Models," CVPR 2024

#### 背景：扩散模型推理的特殊性

扩散模型（Stable Diffusion 等）的推理是**多步迭代去噪**过程（通常 20–50 步），而非 LLM 那样的自回归序列生成：

```
随机噪声 x_T
    ↓ 步骤 T 的 U-Net（或 DiT）前向传播
x_{T-1}
    ↓ 步骤 T-1 的前向传播
x_{T-2}
    ↓ ...（重复 N 步）
清晰图像 x_0
```

每步的前向传播**完全相同**（用同一个模型权重，输入是当前的中间噪声状态）。问题：能否将单步内的计算分布到多 GPU 上，实现 Intra-Step 并行？

**挑战**：Stable Diffusion 的 U-Net 包含 Self-Attention 层，Self-Attention 需要**全局感受野**（每个位置需要看到所有其他位置）——如果将图像划分到不同 GPU，每个 GPU 只有图像的一部分，Self-Attention 无法在不通信的情况下完成。

#### DistriFusion 的核心思路：Asynchronous Displacement（异步位移）

**关键洞察**：相邻时间步 $t$ 和 $t-1$ 的激活值（中间特征图）**变化极小**（扩散过程是渐进的）：

$$|h^{(t)} - h^{(t-1)}|_2 \approx 0.01 \times |h^{(t)}|_2$$

因此，在步骤 $t$ 执行时，用**上一步 $t-1$ 的激活值**做 AllGather（通信）是安全的——精度损失极小，但通信可以完全与计算重叠！

**实现方案：**

```
步骤 t-1 开始前：
  AllGather 步骤 t-2 的激活值（AllGather 在后台异步进行）

步骤 t-1 执行中：
  Convolution 层：使用本地图像块（无需通信）
  Self-Attention 层：使用上一步 t-2 的全局激活值（已 AllGather 完成）
  ↑ 计算和通信完全重叠！

步骤 t-1 结束时：
  将本 GPU 的激活值发布，供步骤 t 的 AllGather 使用
```

**精度分析：**

- 步骤 0 时，上一步的激活值不存在 → 只运行一次完整的非并行步骤做"热身"
- 此后的所有步骤都使用一步前的激活值做 Attention → 微小精度损失，在 FID（Fréchet Inception Distance）指标上几乎不可见

**性能结果（DiT-XL/2，512×512，50 步）：**

- 1 GPU → 4 GPU：加速 **3.6×**（接近线性）
- 通信开销：约占总时间的 5%（被计算完全隐藏）

**更广泛的含义**：DistriFusion 的思路（利用相邻步骤的激活值相似性实现异步通信）是"Staleness 容忍"（Stale Synchrony）在推理场景中的应用，与训练中的 Hogwild! 和异步 SGD 的思想相通。

---

## Topic C：通信综合与优化

### C.1 Collective Communication 算法综合

> **核心论文**：Cai et al., "Synthesizing Optimal Collective Algorithms," PPoPP 2021（MSCCL）

#### 背景：为什么需要算法综合？

传统 NCCL 提供了固定的 AllReduce、AllGather 等算法实现（通常是 Ring 或 Tree）。但不同硬件拓扑（NVLink Switch、IB Fat-Tree、Torus）和不同问题规模下，最优算法完全不同：

- 小数据量（< 1 MB）：延迟主导，Tree 算法更好（$O(\log P)$ 步而非 $O(P)$）
- 大数据量（> 100 MB）：带宽主导，Ring AllReduce 更好（带宽利用率接近 100%）
- 特定拓扑（如 2D Torus）：定制算法才能充分利用拓扑带宽

**问题**：人工设计每种拓扑 × 问题规模 × Collective 类型的最优算法组合是指数级困难的。

#### MSCCL 的自动综合方法

**形式化建模**：将 Collective Communication 算法建模为一个**时间表（Schedule）**：在每个时间步，每个 GPU 执行哪些 Send/Recv 操作，对接收的数据执行哪些 Reduce 操作：

```
AllReduce 的时间表（P=4 GPU 的 Ring，简化）：
时间步 1：GPU0 向 GPU1 发送 chunk_A；GPU1 向 GPU2 发送；...（形成环）
时间步 2：GPU1 累加 chunk_A；向 GPU2 发送累加后的 chunk_A；...
...
时间步 2P-2：所有 GPU 拥有完整 AllReduce 结果
```

**综合目标**：找到满足正确性约束（所有 GPU 最终拥有 AllReduce 结果）的时间表，使总通信时间最小。

**约束规划（Constraint Synthesis）**：

MSCCL 将时间表的搜索建模为一个**约束满足问题（CSP）**，用 Z3 等 SMT Solver 求解：

1. 决策变量：每个时间步内，每对 (sender, receiver) 是否传输哪个 chunk 的哪个部分
2. 正确性约束：数据依赖（必须先收到才能 Reduce 或发送）、最终正确性（所有节点获得完整 AllReduce）
3. 容量约束：每条链路每时间步最多传输一个 chunk
4. 目标：最小化完成所有传输的时间步数

**结果**：对于特定拓扑（如 DGX-2 的 16 GPU NVLink 拓扑），综合出的 AllReduce 算法比 NCCL 默认算法快 **1.5–2×**，尤其在中等数据量（10–100 MB）时优势显著。

**工程实现**：MSCCL 生成的算法以 XML 格式描述，由 MSCCL Runtime 动态调度，与 NCCL API 完全兼容（用户无需修改代码）。

### C.2 Breaking the Computation-Communication Abstraction Barrier

> **论文**：Jangda et al., "Breaking the Computation and Communication Abstraction Barrier in Distributed Machine Learning Workloads," ASPLOS 2022（CoCoNet）

#### 问题：计算-通信的抽象壁垒

传统分布式 ML 框架将"计算"和"通信"视为独立的抽象层：

```
用户代码：
  result = allreduce(tensor)   # 通信
  output = relu(result)        # 计算
            ↕（清晰分离的界面）
系统层：
  NCCL 负责 allreduce
  cuBLAS/Triton 负责 relu
```

这种分离使系统无法进行**跨越计算-通信界面**的优化，例如：

- 在 AllReduce 传输 chunk A 的同时，对已接收的 chunk B 执行 ReLU（部分计算与通信重叠）
- 将一个大 Tensor 的 AllReduce 和后续的 LayerNorm 融合，避免中间 Tensor 写回 HBM

#### CoCoNet 的方法：统一计算-通信图

CoCoNet 将通信原语（Send/Recv/AllReduce）和计算原语（MatMul/ReLU/LayerNorm）表示在**同一个计算图 IR** 中，并进行联合优化：

**转换 1：通信操作的 Chunk 化**

将一次 AllReduce 拆分为多个 Chunk 的 AllReduce，使后续的计算可以在部分 Chunk 完成通信后立即开始：

```
传统：
AllReduce(全部梯度) → ReLU(全部梯度)   # 必须等 AllReduce 完全结束

CoCoNet：
AllReduce(Chunk 0) → ReLU(Chunk 0)   # 立即开始
AllReduce(Chunk 1) → ReLU(Chunk 1)   # Chunk 0 的 ReLU 在进行
...                                   # 深度重叠
```

**转换 2：通信与计算的 Kernel Fusion**

将一次 AllReduce 的最后 Reduce 步骤与后续的 LayerNorm 融合为单个 Kernel：

```
传统：
AllReduce_last_step(写 HBM) + LayerNorm(读 HBM)

CoCoNet：
Fused_AllReduce_LayerNorm(AllReduce 结果留在寄存器/Shared Mem，直接 LayerNorm)
```

**实验结果（GPT-3 训练，DGX-A100 集群）**：在通信量较大的配置下，CoCoNet 将通信开销减少约 **35%**，训练吞吐提升约 **15–30%**。

---

## Topic D：专用加速器

### D.1 TPU：数据流与脉动阵列架构

> **核心论文**：Jouppi et al., "TPU v4: An Optically Reconfigurable Supercomputer for Machine Learning," ISCA 2023
> **背景**：Jouppi et al., "In-Datacenter Performance Analysis of a Tensor Processing Unit," ISCA 2017（原始 TPU 论文）

#### 从 GPU 到 TPU：为什么需要专用加速器？

GPU 是为图形渲染设计的通用并行处理器，ML 只是后来的应用。它有很多对 ML 非必要的硬件（纹理单元、光栅化器、几何着色器）和对 ML 性能不利的设计（为低延迟优化、不规则控制流支持）。

**ML 工作负载的特征**（Google 2015 年分析 Google 数据中心的 ML 任务）：

- 95% 的 ML 计算是**矩阵-向量乘法**（Dense GEMV）
- 批量大小通常很小（推理场景 batch=1–16）
- 几乎没有复杂控制流
- 对精度要求低（INT8 甚至更低即可满足推理需求）

这些特征促使 Google 在 2015 年启动 TPU 项目——一个**专为矩阵乘法优化的 ASIC**。

#### 脉动阵列（Systolic Array）：TPU 的核心计算单元

脉动阵列是 TPU 的心脏，专为高效执行矩阵乘法设计：

```
脉动阵列（4×4 示意，实际 TPU 为 256×256）：

      B[0,0] B[0,1] B[0,2] B[0,3]  ← 权重矩阵 B 按列流动（→）
        ↓      ↓      ↓      ↓
A[0] → PE  →  PE  →  PE  →  PE  →  C[0,:] 部分和
         ↓      ↓      ↓      ↓
A[1] → PE  →  PE  →  PE  →  PE  →  C[1,:] 部分和
         ↓      ↓      ↓      ↓
A[2] → PE  →  PE  →  PE  →  PE  →  C[2,:] 部分和
         ↓      ↓      ↓      ↓
A[3] → PE  →  PE  →  PE  →  PE  →  C[3,:] 部分和

PE（Processing Element）：执行一次 MAC（乘加）操作
A 矩阵的行从左侧流入（↓），B 矩阵的列从顶部流入（→）
每个 PE 将接收的 A 和 B 相乘并累加到本地寄存器
```

**脉动阵列的关键特性：**

1. **数据流（Dataflow）驱动**：数据在 PE 阵列中流动，每个数据从一个 PE 流向下一个 PE，无需每次访问中央内存（对比 GPU 的 Cache 层次）
2. **极高数据复用**：矩阵 B 的每个元素被复用 N 次（N = A 的行数）；矩阵 A 的每个元素也被复用 N 次。256×256 的脉动阵列，每加载一次数据可以执行 65536 次 MAC
3. **完全可预测的执行时序**：数据的流动路径和时序完全确定，无投机执行、无 Cache Miss 不确定性

**脉动阵列 vs GPU 的计算层次差异：**

```
GPU（NVIDIA A100）计算路径：
HBM → L2 Cache → L1 Cache / Shared Memory → Register → Tensor Core → Register → 写回

TPU 脉动阵列计算路径：
HBM → 权重缓冲区（Weight Stationary，可选）→ 脉动阵列 → 累加器寄存器 → 写回
              ↑ 只需一次从 HBM 读取权重
```

**为什么脉动阵列对矩阵乘法高效**：权重矩阵 $W$ 只需从内存读取一次，可以被所有输入向量复用（Weight Stationary Dataflow）——这对 LLM 推理中权重远大于激活的场景极其有效。

#### TPU v1 到 TPU v4 的演进

**TPU v1（2015–2017）：**

- 仅推理，INT8
- 256×256 脉动阵列 × 1 核
- 92 TOP/s（INT8）
- 8 GB HBM，30 GB/s 带宽（很低！）
- 主要瓶颈：内存带宽

**TPU v2/v3（2017–2019）：**

- 支持训练，BF16
- 引入 HBM，大幅提升带宽
- v3：420 TFLOP/s（BF16），900 GB/s HBM

**TPU v4（2023，ISCA 论文）：**

TPU v4 的最重要创新是**光学可重构互联（Optically Reconfigurable Interconnect）**：

```
传统超级计算机互联：
  芯片 A —— 固定铜缆 —— 芯片 B
  （拓扑一旦安装就固定，若某节点故障则整块区域不可用）

TPU v4 光学互联：
  芯片 A ←→ [光学电路交换机（OCS）] ←→ 芯片 B/C/D/...
  （OCS 可以在毫秒级重新配置连接拓扑！）
```

**光学可重构互联的好处：**

1. **容错**：某个 TPU 芯片故障时，OCS 在数毫秒内重新路由绕过故障节点，训练任务无需重启
2. **拓扑优化**：根据当前正在训练的模型（不同的通信模式），动态调整互联拓扑（如 AllReduce 需要 Ring，Embedding 访问需要特定切分）
3. **规模**：4096 个 TPU v4 芯片构成一个 TPU Pod，峰值算力 ~1.1 EFLOP/s（BF16，Exaflop 级别）

**TPU v4 的嵌入式 HBM（Embedding）支持：**

广告推荐等 Embedding 密集型应用中，超大 Embedding 表（数 TB）无法放进 HBM，TPU v4 提供了对 DRAM 和 SSD 的直接访问路径（绕过 PCIe）。

#### TPU 的局限性

- **灵活性差**：脉动阵列对非矩阵乘法运算（如 Softmax、Layer Norm 等逐元素操作）效率低
- **稀疏计算困难**：MoE 的稀疏门控对 TPU 的静态调度不友好
- **动态形状支持差**：XLA 编译器擅长静态形状，动态形状（如变长序列）需要 Padding，浪费计算
- **仅 Google 内部（TPU v4/v5）**：无法通过云 API 完整控制硬件

---

### D.2 Groq TSP：确定性张量流式处理器

> **论文**：Abts et al., "Think Fast: A Tensor Streaming Processor (TSP) for Accelerating Deep Learning Workloads," ISCA 2020

#### 设计哲学：确定性（Determinism）优先

GPU 和 CPU 的性能波动来源很多：Cache Miss 的随机性、OOO 乱序执行的动态性、内存访问竞争。Groq TSP 的设计目标是**消除所有不确定性**——每条指令的执行时间在编译时就精确已知，整个程序的执行时序是确定的（Deterministic）。

**确定性的实现方式：**

1. **无 Cache**：彻底取消硬件 Cache，改用显式管理的 Scratchpad Memory（编译器决定数据在何时位于何处）
2. **In-order Execution（顺序执行）**：无 OOO，无推测执行，每条指令在固定的时钟周期执行
3. **静态调度（Static Scheduling）**：编译器（而非硬件调度器）决定每个周期执行哪条指令，发射给哪个功能单元
4. **超长指令字（VLIW-like）**：每个时钟周期，编译器同时控制多个独立的功能单元（类似 VLIW 处理器）

#### TSP 的架构

```
TSP 芯片布局（概念性）：

Slice 0  | Slice 1  | Slice 2  | ... | Slice N
[VXM][MXM][VXM][MXM][VXM][MXM]...[VXM][MXM]
[───────────── SRAM ───────────────────────]
[───────── Super Lane Network ─────────────]

VXM（Vector eXecution Module）：向量 ALU，处理 Elementwise 操作
MXM（Matrix Multiply Module）：矩阵乘法单元，对应脉动阵列
SRAM：每个 Slice 有本地 SRAM，数据在 Slice 间的 Super Lane Network 上流动
```

**Tensor Streaming（张量流式处理）**：

TSP 的核心执行模型是"流"——数据像流水线一样流过 MXM 和 VXM 单元，而不是像 GPU 那样的"计算 + 写回 + 读取"反复循环：

```
矩阵乘法 + ReLU 的执行（无 Cache，无写回中间结果）：
  数据从 SRAM →（Super Lane）→ MXM（矩阵乘法）→（流动到）→ VXM（ReLU）→ 写回 SRAM
  ↑ 全程数据在片内流动，不落到 HBM
```

**确定性使什么成为可能：**

当所有延迟都精确已知，编译器可以：

- 精确安排数据传输（在数据被需要之前恰好几个周期触发加载）
- 完全消除流水线停顿（无 Cache Miss 不确定性）
- 实现理论上的 100% 功能单元利用率（不存在等待）

#### Groq 与 GPU 的性能对比

|指标|NVIDIA A100|Groq LPU（单芯片）|
|---|---|---|
|峰值算力|312 TFLOP/s（FP16）|188 TFLOP/s（FP16）|
|**推理延迟（LLaMA-3-70B，单 token）**|~200 ms（1 GPU）|**~5 ms**（8 芯片）|
|多 token 吞吐（1 GroqRack = 8 芯片）|约 1000 tok/s|约 800 tok/s|

**Groq 的核心优势是延迟**，而非吞吐。由于完全消除了 Cache Miss 和调度不确定性，Groq 可以在极低延迟下完成推理，适合延迟敏感的实时场景。

#### TSP 的局限性

- **编译器复杂度极高**：静态调度要求编译器解决 NP 难的调度问题（在 100% 确定性的约束下）
- **动态形状不友好**：完全静态的调度无法处理运行时才知道大小的张量（如变长序列的 KV Cache）——需要按最大长度 Pad
- **通用性差**：对于脉动阵列不擅长的稀疏计算、不规则控制流，TSP 无优势
- **编程模型受限**：用户无法用 CUDA/Triton 等通用语言编程，只能通过 Groq 编译器

---

## Topic E：AI Scaling

### E.1 Scaling Laws：神经缩放律

> **核心论文**：Kaplan et al., "Scaling Laws for Neural Language Models," 2020

#### 幂律关系（Power Law）

Kaplan 等人通过系统性实验（训练数百个不同规模的 Transformer）发现，语言模型的测试损失（Test Loss）与三个变量之间存在**幂律（Power Law）关系**：

$$L(N) \approx \left(\frac{N_c}{N}\right)^{\alpha_N}, \quad L(D) \approx \left(\frac{D_c}{D}\right)^{\alpha_D}, \quad L(C) \approx \left(\frac{C_c}{C}\right)^{\alpha_C}$$

其中：

- $N$：模型参数量
- $D$：训练数据量（Token 数）
- $C$：计算量（FLOPs）
- $\alpha_N \approx 0.076$，$\alpha_D \approx 0.095$，$\alpha_C \approx 0.050$（Kaplan 2020 的估计）

**幂律的核心含义**：

1. **平滑性**：Loss 随规模的增长是**平滑、可预测的**（不存在明显的相变点），模型越大、数据越多、计算量越大，性能都稳定提升
2. **预测能力**：通过在小模型上观测到的 Scaling 趋势，可以**外推预测**更大模型的性能——这是大模型训练预算规划的基础
3. **规律的普遍性**：不同架构（Transformer decoder、encoder、encoder-decoder）、不同领域（代码、中文、图像 token）都遵循类似的幂律

**固定计算量 $C$ 下的最优分配**（Kaplan 2020 的关键结论）：

给定一个计算预算 $C$（以 FLOPs 计），如何分配到模型大小 $N$ 和数据量 $D$？Kaplan 发现：

> 在给定计算量下，**优先扩大模型规模**，数据量适当增加即可。

具体而言：最优的 $N$ 与 $C$ 的 $0.73$ 次方成正比，而最优的 $D$ 与 $C$ 的 $0.27$ 次方成正比——即模型规模增长应该远快于数据量增长。

这一结论促成了 GPT-3（175B 参数，相对少量数据训练）等大而"数据欠训练"模型的盛行。

---

### E.2 Chinchilla：计算最优训练

> **论文**：Hoffmann et al., "Training Compute-Optimal Large Language Models," NeurIPS 2022

#### 对 Kaplan 的修正

Hoffmann 等人（DeepMind）指出 Kaplan 2020 的实验设计存在缺陷：**小模型训练不足**（训练步数固定，小模型达到收敛而大模型还未收敛），导致低估了数据的重要性。

重新实验后，Chinchilla 得出的最优分配为：

$$N_{opt} \propto C^{0.49}, \quad D_{opt} \propto C^{0.51}$$

**即模型参数量和训练 Token 数应该大致等比例增长，而非 Kaplan 建议的参数量增长更快。**

**Chinchilla 的经验法则：**

$$D_{opt} \approx 20 \times N$$

对于 $N$ 个参数的模型，应训练约 $20N$ 个 Token 才能达到计算最优（Compute-Optimal）。

**Chinchilla 验证**：DeepMind 用与 Gopher（280B 参数）相同的计算量，训练了一个更小但数据更多的模型：

|模型|参数量|训练 Token 数|性能（各基准平均）|
|---|---|---|---|
|Gopher|280B|300B tokens|基准|
|Chinchilla|70B|**1.4T tokens**|**优于 Gopher**|

Chinchilla（70B + 1.4T tokens）在所有评估基准上优于 Gopher（280B + 300B tokens），且推理成本仅为 Gopher 的 1/4。

#### Chinchilla 对业界的冲击

这一发现迅速改变了大模型的训练策略：

- **LLaMA（Meta，2023）**：7B/13B/65B 参数，但每个模型都训练了超过 1T token（远超 Chinchilla 比例）。结果是 LLaMA-13B 在多数基准上超过 GPT-3（175B），因为 GPT-3 训练数据远少于 Chinchilla 最优
- **Mistral 7B（2023）**：7B 参数 + 约 1T token 训练，性能超过 LLaMA-13B
- **DeepSeek 系列**：始终强调"Compute-Optimal"和"Token-Optimal"训练

**关键洞察**：如果推理时的服务成本（用户请求数 × 推理成本）在模型生命周期内足够高，那么用**更多数据训练更小的模型**是经济最优的——训练时多花一些计算，换取每次推理成本的大幅降低。

---

### E.3 推理侧 Scaling 与新范式

#### Inference-Time Compute Scaling（推理时计算 Scaling）

Kaplan 和 Chinchilla 的 Scaling Laws 都聚焦于**训练时的计算**。2024 年以来，"推理时 Scaling"（Test-Time Compute Scaling）成为新方向：

**核心思想**：给模型**更多推理时间**（更多计算 FLOPs），可以提升特定任务的性能。

**主要实现方式：**

**Chain-of-Thought（思维链）**：让模型生成中间推理步骤，而非直接给出答案。增加生成 Token 数 = 增加推理计算量：

```
直接回答（少推理计算）：
Q: "2+2 等于多少的 100 次方？" → "很大的数字"（可能错）

思维链（多推理计算）：
Q: "2+2 等于多少的 100 次方？"
A: "首先，2+2=4。然后，4^100 = 4^100。
    4 = 2^2，所以 4^100 = 2^200。
    2^10 ≈ 10^3，所以 2^200 ≈ (2^10)^20 ≈ 10^60。
    答案约为 10^60。"（正确方向）
```

**Best-of-N Sampling**：对同一问题生成 N 个答案，选择最优（通过 Reward Model 打分）。成本 = N × 单次推理成本，性能随 N 对数增长。

**Search-based Methods（MCTS / Beam Search）**：在生成树上做搜索，探索多条路径并选择最高分。OpenAI o1/o3 和 DeepSeek-R1 使用了这类机制。

**推理时 Scaling 的 Scaling Law**：

$$\text{Performance} \propto C_{inference}^{\alpha_{infer}}$$

实验表明 $\alpha_{infer}$ 约为 0.3–0.5，比训练时 Scaling（$\approx 0.05$）陡峭——即推理计算更"物美价廉"（每增加 1× 计算，性能提升更多）。

#### 推理 Scaling 对系统的含义

**o1/o3 风格模型的系统挑战：**

- 单次查询可能生成数千到数万 Token（vs 普通对话的几百 Token）
- KV Cache 大幅增长（更长的 Context）
- 需要在**Latency 和 Throughput 之间**找新的平衡（用户愿意等更长时间换取更好的答案）
- Speculative Decoding（推测解码）变得更重要：用小模型快速生成候选 Token，大模型批量验证

---

### E.4 Scaling 的基础设施含义

大规模训练和推理的 Scaling 对基础设施提出了极端要求：

#### Tectonic-Shift：大规模 ML 存储（Meta，ATC 2023）

**规模**：Meta 在 2021 年的训练集群产生了 ~1 exabyte 的数据（检查点 + 训练数据）。传统 HDFS 无法满足 ML 工作负载的特殊需求：

|需求|传统 HDFS 设计|ML 工作负载特征|
|---|---|---|
|读取模式|顺序、一次性|训练数据多 Epoch 重复顺序读取|
|写入模式|偶发、小量|Checkpoint 时突发大量写入（350GB+ 一次）|
|文件大小|大文件为主|混合：从 1 KB 样本文件到 100 GB+ 模型文件|
|访问延迟要求|高延迟可接受|需要与 GPU 计算速度匹配（否则成为瓶颈）|

**Tectonic-Shift 的主要优化：**

1. **Foreground Buffering（前台缓冲）**：Checkpoint 写入先进入快速 SSD 缓冲层，异步刷入 HDD，避免 Checkpoint 阻塞训练（传统：同步写入慢速 HDD = 训练暂停 10+ 分钟）
2. **Tiered Storage（分层存储）**：热数据（近期 Checkpoint、正在使用的训练集）放 SSD；冷数据（历史 Checkpoint、归档数据集）放 HDD 或对象存储
3. **Read-Ahead Prefetch（预读预取）**：检测 ML 训练的顺序访问模式，提前预取下 N 个数据文件，使 GPU 数据加载不再是瓶颈

#### MAST：全球 ML 训练调度（Meta，OSDI 2024）

**问题**：Meta 在全球多个地理位置运营数据中心，ML 训练作业需要数千 GPU 运行数周。任何设施的局部故障（网络分区、单个数据中心断电）都可能导致整个训练作业失败。

**MAST 的核心机制：**

1. **全球视角调度（Global-View Scheduling）**：一个中央调度器知道所有数据中心的 GPU 状态（可用、忙碌、故障），为每个训练作业选择最优的 GPU 集合（考虑网络拓扑、故障历史、剩余容量）
    
2. **Goodput 最大化**：调度目标不是最大化 GPU 利用率，而是最大化 **Goodput**（有效完成的训练步数 / 时间）：
    
    - 预测每个候选资源配置的预期 Goodput（考虑故障概率）
    - 优先选择故障率低的节点，即使暂时 GPU 利用率不是 100%
3. **快速故障恢复**：维护多个 Shadow Job（影子作业），一旦主作业节点故障，立即用 Shadow Job 从最近 Checkpoint 重启（Warmup 时间 < 1 分钟）
    

#### Singularity：弹性 + 抢占式调度（Microsoft，2022）

**设计目标**：在数万 GPU 的生产集群中，支持 ML 训练作业的**动态抢占**和**弹性调整**，不需要作业重启：

- **抢占（Preemption）**：高优先级作业（如推理服务流量突发）需要 GPU 资源时，可以暂停正在训练的低优先级作业
- **弹性（Elasticity）**：允许作业在运行中动态增加或减少 GPU 数量（如集群有空闲 GPU 时自动扩展）

**技术挑战**：分布式 ML 训练中，所有 GPU 是紧耦合的（AllReduce 等待所有参与者）。若某个 GPU 被抢占，整个作业就停顿。

**Singularity 的解决方案**：在 ML 框架和硬件之间插入一个**虚拟化层（Execution Substrate）**：

1. 定期自动创建**轻量级 Checkpoint**（比传统 Checkpoint 小 10×，通过只存储 diff 实现）
2. 接到抢占信号时，等到下一个 Checkpoint 点（通常 < 30 秒），保存状态后暂停
3. 资源恢复后，从 Checkpoint 无缝恢复（用户感知 = 训练速度略慢）
4. 弹性扩展：新增 GPU 加入时，重新分片数据并行组，从当前 Checkpoint 继续（支持 FSDP 的动态 resharding）

---

## Topic F：自动微分的关系代数视角

> **论文**：Tang et al., "Auto-Differentiation of Relational Computations for Very Large Scale Machine Learning," ICML 2023

### 背景：为什么需要关系代数视角？

Part II 的 Module 11 介绍了标准的 Reverse Mode AD（基于 Tape / 动态图）。它对 Tensor 操作非常高效，但对于**关系型计算**（数据库查询、稀疏 Embedding 访问、图神经网络的邻居聚合）效率低下：

```python
# 标准 ML：Tensor 操作
output = W @ x    # 矩阵乘法，所有元素稠密
grad_W = output_grad @ x.T   # 高效反向传播

# 关系型计算（如推荐系统 Embedding）：
output = embedding_table[user_ids]  # 选取特定行（极稀疏）
# 梯度：需要将 output_grad 散回 embedding_table 的对应行
# 标准 AD 对此非常低效（因为 embedding_table 可能有 10^9 行）
```

对于 Meta 的广告推荐模型，Embedding Table 可能有数百 GB，包含数十亿行。每次前向传播只访问其中少数行（1000 个用户），标准 AD 的 Tape 无法高效处理这种极端稀疏的梯度。

### 关系代数视角的贡献

**核心思想**：用**关系代数（Relational Algebra）**的视角重新表示 ML 程序，将 ML 计算中的 Join、GroupBy、Select 等操作识别出来，然后推导这些关系操作的**代数性微分规则**（类似于 Tensor 操作有 Chain Rule）：

```
ML 操作（Tensor 视角）：
  output = embedding[user_id]   # 按索引选取行

关系代数视角：
  output = π_cols(σ_{id=user_id}(embedding))
  其中 π = Projection，σ = Selection
  
关系代数的微分规则（类比 Chain Rule）：
  ∂L/∂embedding = scatter(∂L/∂output, user_id)
  ↑ 只需更新被访问的稀疏行，其余行梯度为零
```

**实验结果**：在超大规模推荐模型（Embedding Table 数千亿参数）上，比标准 PyTorch AD 快 **5–10×**，内存占用减少 **2–4×**。

**对课程的重要性**：这篇论文展示了 ML 系统研究的一个重要方向——将 CS 其他领域（数据库、编译器）的成熟理论引入 ML 系统，而不是重新发明轮子。

---

## Part III 总结：横跨三部分的系统思维框架

### 全课程知识图谱

```
Part I：基础层（"What does hardware do?")
  CPU（流水线、乱序、SMT）
  GPU（SIMD、Warp、Tensor Core）
  Memory（SRAM/DRAM/HBM、Cache 层次）
  数据表示（浮点、整数、稀疏格式）
        ↓ 为 Part II 提供硬件性能模型
        
Part II：系统层（"How do ML programs run efficiently?")
  计算图优化（算子融合、并行策略）
  Loop 变换（Tiling、Unrolling、Software Pipeline）
  Roofline 模型（判断瓶颈、指导优化方向）
  内存优化（量化、稀疏、Checkpointing）
  通信（Ring AllReduce、Collective Communication）
  训练（AD、梯度下降、DP/PP/TP/FSDP/ZeRO）
  微调（LoRA、Adapter）
  MoE（稀疏激活、负载均衡、推理优化）
        ↓ 为 Part III 提供系统设计基础
        
Part III：前沿层（"What are people building and why?")
  代码生成（Triton、PyTorch 2 / TorchDynamo）
  推理服务（vLLM / PagedAttention、量化推理）
  分布式推理（DistriFusion、CoCoNet）
  通信优化（MSCCL / 算法综合）
  专用加速器（TPU / 脉动阵列、Groq / TSP）
  Scaling Laws（Kaplan、Chinchilla、推理 Scaling）
  大规模基础设施（Tectonic、MAST、Singularity）
  关系代数 AD（Embedding 梯度的新范式）
```

### 所有优化手段的统一框架

回顾 Compendium 的框架：所有优化归结为三件事。

**减少 Work（$W$）：**

|技术|应用层次|
|---|---|
|量化（FP32→INT8/INT4）|推理内存流量 ÷ 4|
|稀疏化（MoE、权重剪枝）|激活参数量 ÷ n|
|算法改进（FlashAttention）|Attention 的内存流量 $O(N) \to O(N^2/B)$ |
|梯度压缩|通信量 ÷ 10–100|

**降低 Cost（$t$）：**

|技术|应用层次|
|---|---|
|Loop Tiling|Cache Hit Rate ↑，HBM 访问 ↓|
|向量化 / Tensor Core|每条指令处理更多数据|
|算子融合|消除 HBM 中间写入|
|低精度（FP16/BF16）|存储量 ÷ 2，Tensor Core 速度 × 2|

**增加 Parallelism（$P$）：**

|技术|应用层次|
|---|---|
|Data Parallelism|Batch 级并行|
|Pipeline Parallelism|层级并行（以 Bubble 为代价）|
|Tensor Parallelism|算子内并行（高带宽要求）|
|FSDP / ZeRO|打破"模型要求单 GPU 能装下"的限制|
|Triton / TorchInductor|自动化提升单 Kernel 并行度|

### 对你的 Profiler 项目的完整视角

经过三部分的学习，你的 LLM Quantization Tax Profiler 在课程框架中的位置已经很清晰：

1. **Part I 基础**：INT4 的位运算表示（Module 4）；HBM 带宽和 RTX 4060 的内存层次（Module 5）
2. **Part II 核心**：Roofline 模型（Module 8）——profiler 测量的就是 kernel 在 Roofline 图上的位置；量化对算术强度的影响（Module 9）
3. **Part III 具体**：Marlin INT4 GEMM 的 batch-size 行为就是 B.2 节分析的内容；你的 profiler 在实测层面验证了 QMoE 和 Marlin 论文的系统级分析

**Profiler 项目与 Pai 老师的研究方向（Roofline + MLSys）完美契合**，这是你在课程开始时最应该在 Office Hours 建立联系的切入点。

---

文档基于 Sreepathi Pai，CSC 290/420 ML Systems for Efficient AI，University of Rochester，[Fall 2025 课程网站](https://www.cs.rochester.edu/~spai4/courses/csc-290-420/fall-2025/)及 [Fall 2024 论文列表](https://www.cs.rochester.edu/u/sree/courses/csc-290-571/fall-2024/schedule.html)整理。