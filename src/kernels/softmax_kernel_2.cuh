#pragma once  // 包含守卫：同一翻译单元内只编译一次


#include <cuda_runtime.h>   // CUDA Runtime API：dim3 / __global__ / __expf / fmaxf /
                             //   threadIdx / blockIdx / __syncthreads 等 kernel 编写所需
// BLOCK_DIM_Y：block 在 y 维度的线程数，决定 shared memory 大小和每行的并行线程数
//   本文件不定义此宏，由调用方（run_kernels.cu）在 include 前通过以下方式提供：
//     #ifndef BLOCK_DIM_Y
//     #define BLOCK_DIM_Y 1024
//     #endif
//   或通过编译器 -DBLOCK_DIM_Y=1024 注入（CMakeLists.txt 的 target_compile_definitions）

#ifndef BLOCK_DIM_X
#define BLOCK_DIM_X 1024
#endif

template <typename scalar_t,  typename scalar_i>
__global__ void softmax_kernel_2(scalar_t* __restrict__ a, scalar_t* __restrict__ b, scalar_i totalRow, scalar_i totalCol) {
  scalar_i row = blockIdx.x * blockDim.x + threadIdx.x;
  scalar_i ty = threadIdx.y;
  if (row < totalRow) {
    __shared__ float reduction[BLOCK_DIM_X];
    float maxval = -INFINITY;
    for (scalar_i i = ty * BLOCK_DIM_X; i<min(totalCol, (ty+1) * BLOCK_DIM_X); i+=1)
    {
      maxval = fmaxf(maxval, a[row*totalCol + i]);
    }

    reduction[ty] = maxval;
    for(scalar_i stride = BLOCK_DIM_X/2; stride>=1; stride/=2)
    {
      __syncthreads();
      if (ty < stride)
      {
        reduction[ty] = fmaxf(reduction[ty], reduction[ty+stride]);
      }
    }

    __syncthreads();
    maxval = reduction[0];
    float divisor = 0.f;
    for (scalar_i i = ty * BLOCK_DIM_X; i<min(totalCol, (ty + 1) * BLOCK_DIM_X); i+=1)
    {
      divisor += __expf(a[row * totalCol + i] - maxval);
    }
    reduction[ty] = divisor;
    for(scalar_i stride = BLOCK_DIM_X/2; stride>=1; stride/=2)
    {
      __syncthreads();
      if (ty < stride)
      {
        reduction[ty] = reduction[ty] + reduction[ty+stride];
      }
    }
    __syncthreads();
    divisor = reduction[0];

    for (scalar_i i = ty; i<totalCol; i+=BLOCK_DIM_X)
    {
      b[row*totalCol + i] = __expf(a[row*totalCol + i]-maxval)/divisor;
    }
  }
}




// ─── 为什么用树形规约能提高 GFLOPS ───────────────────────────────────────────
//
// GFLOPS = 浮点运算次数 / 执行时间。
// 总操作数固定（7n × m），要提高 GFLOPS 只能缩短执行时间。
//
// 【串行规约的瓶颈】
//   若用单线程对一行 n 个元素求 max / sum：
//     时间 ∝ n（顺序依赖，无法并行）
//   其余 k-1 个线程全部空闲，SM 的算力和内存带宽严重浪费。
//   空闲线程 → warp 占用率低 → 无法用其他 warp 的指令掩盖内存延迟（latency hiding 失效）。
//
// 【树形规约的改进】
//   Step 1  串行阶段：k 个线程各自负责 n/k 列，并行执行，时间 ∝ n/k。
//   Step 2  规约阶段：log₂(k) 轮，每轮活跃线程减半，时间 ∝ log₂(k)。
//   总时间 ∝ n/k + log₂(k)，远小于串行的 n。
//
//   示例（k = BLOCK_DIM_X = 1024，n = 4096）：
//     串行时间 ∝ 4096
//     树形时间 ∝ 4096/1024 + log₂(1024) = 4 + 10 = 14   → 约 293× 加速
//
// 【附带收益】
//   · k 个线程并发发出内存请求 → 更充分利用 HBM/L2 带宽。
//   · 更多活跃 warp → scheduler 可在内存延迟期间切换执行其他 warp（latency hiding）。
//
// 【代价】
//   · 需要 shared memory（BLOCK_DIM_X × 4 字节）存储各线程的中间结果。
//   · 每轮规约需要 __syncthreads()，引入同步开销（共 log₂(k) 次）。
//   · 当 n ≪ k 时，大量线程空转，规约收益下降。
// ─────────────────────────────────────────────────────────────────────────────
template <typename scalar_t,typename scalar_i>
__global__ void softmax_kernel_reduction(scalar_t* __restrict__ a, scalar_t* __restrict__ b, scalar_i totalRow, scalar_i totalCol) {
  // 该block负责的启始行
  scalar_i initRow = blockIdx.y * blockDim.y;
  // 该线程的起始行
  scalar_i row {threadIdx.y};
  // 该线程负责的列组
  scalar_i colGroup {threadIdx.x};
  // 该block的线程数
  scalar_i threadNum {blockDim.x};


  // Bug 4（非 bug）：__syncthreads() 位于条件分支内，通常不安全（半个 block 进入会死锁）
  // 此处安全的原因：blockDim.y=1 → threadIdx.y 恒为 0 → row + initRow = blockIdx.y 对块内所有线程相同
  // 条件结果全块一致：要么全部进入，要么全部跳过，不存在分歧，__syncthreads() 不会死锁
  if (row + initRow < totalRow) {

    // 该block静态SMEM
    // 一个block的线程要么启动，要么全部启动，所以静态SMEM的分配可以放到if条件判断内
    // 修复：原为 reduction[threadNum]（VLA，运行时值），CUDA 静态 shared memory 必须用编译期常量
    float __shared__ reduction[BLOCK_DIM_X];

    // 每个block中的线程进行串行规约
    // 初始值用 -INFINITY 而非 0.0f：0.0f 在所有输入均为负数时会错误地成为最大值
    float maxval {-INFINITY};
    // 修复：原为 i 从 1 开始且不依赖 colGroup，导致所有线程计算相同元素、遗漏元素 [0]
    // 正确：从 colGroup 开始步长 threadNum，每线程覆盖自己的列组
    for (scalar_i i {colGroup}; i < totalCol; i+=threadNum) {
      maxval = fmax(maxval, a[(row + initRow) * totalCol + i]);
    }
    reduction[colGroup] = maxval;
    // 同步
    __syncthreads();

    // 线程进行树形规约求最大值
    for (scalar_i i {threadNum/2}; i>=1; i/=2) {
      // 每次循环用到的线程减半
      if (colGroup < i) {
        reduction[colGroup] = fmaxf(reduction[colGroup], reduction[colGroup+i]);
      }
      __syncthreads();
    }
    // 最大值规约到位置0
    maxval = reduction[0];
    float sum = 0.f;
    // 线程进行树形规约求和
    // 先存储colGroup列的值
    for (scalar_i col {colGroup}; col < totalCol; col+=threadNum) {
      sum += __expf(a[(row + initRow) * totalCol + col] - maxval);
    }
    // Bug 3（非 bug）：blockDim.y=1 → threadIdx.y 恒为 0 → 块内只有 threadIdx.x（colGroup）变化
    // reduction[colGroup] 的每个槽由唯一线程写入，不存在多线程竞争同一槽的情况
    reduction[colGroup] = sum;
    // 问题2（已修复）：原代码此处缺少 __syncthreads()
    // 所有线程将各自的部分和写入 reduction[] 后必须同步，
    // 确保后续树形规约读取相邻槽时所有写入均已对全块可见
    __syncthreads();
    for (scalar_i i {threadNum/2}; i>=1; i/=2) {
      // 每次循环用到的线程减半
      if (colGroup < i) {
        // Bug 2（已修复）：原代码在求和树形规约中读 a[] 而非 reduction[]
        // 读 a[] 只能取单个原始值，导致其他线程已累积的部分和全部丢失
        // 修复：读 reduction[colGroup+i]，合并右侧线程已累积的部分和
        reduction[colGroup] += reduction[colGroup + i];
      }
      __syncthreads();
    }
    // 求和规约到位置0
    float diversor { reduction[0] };

    // 求这一行的softmax
    // 修复：原为 i 从 0 开始，所有线程写相同位置（0, threadNum, 2*threadNum...），大量列未写入
    // 正确：从 colGroup 开始，每线程写自己负责的列组
    for (scalar_i i {colGroup}; i < totalCol; i+=threadNum) {
      b[(row + initRow) * totalCol + i] = __expf(a[(row + initRow) * totalCol + i] - maxval) / diversor;
    }
  }
}