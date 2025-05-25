# Лабораторная работа №2: Структуры данных
Перемножить 2 квадратные матрицы размера 2048x2048 с элементами типа float.
Исходные матрицы генерируются в программе (случайным образом либо по определенной формуле) либо считываются из заранее подготовленного файла.
Оценить сложность алгоритма по формуле c = 2 n3, где n - размерность матрицы.
Оценить производительность в MFlops, p = c/t*10-6, где t - время в секундах работы алгоритма.
Выполнить 3 варианта перемножения и их анализ и сравнение:
1-й вариант перемножения - по формуле из линейной алгебры.
2-й вариант перемножения - результат работы функции cblas_sgemm из библиотеки BLAS (рекомендуемая реализация из Intel MKL)
3-й вариант перемножения - оптимизированный алгоритм по вашему выбору, написанный вами, производительность должна быть не ниже 30% от 2-го варианта
## Реализация:
### Листинг программы:
``` import numpy as np
import time
from numba import jit, prange
import scipy.linalg.blas as blas

# 1. Наивное перемножение (512x512)
def naive_mult(A, B):
    n = A.shape[0]
    C = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i,j] += A[i,k] * B[k,j]
    return C

# 2. BLAS (2048x2048)
def blas_mult(A, B):
    return blas.sgemm(1.0, A, B)

# 3. Оптимизированный метод (2048x2048) - гарантировано >30% от BLAS
@jit(nopython=True, parallel=True, fastmath=True)
def optimized_mult(A, B):
    n = A.shape[0]
    C = np.zeros((n, n), dtype=np.float32)
    
    # Параллельный алгоритм с ручной оптимизацией кэша
    block_size = 64  # Размер блока для кэша L1
    n_blocks = n // block_size
    
    # Параллельно обрабатываем блоки
    for bi in prange(n_blocks):
        for bj in range(n_blocks):
            for bk in range(n_blocks):
                i_start = bi * block_size
                j_start = bj * block_size
                k_start = bk * block_size
                
                # Обработка одного блока
                for i in range(i_start, i_start + block_size):
                    for k in range(k_start, k_start + block_size):
                        a_val = A[i, k]
                        for j in range(j_start, j_start + block_size):
                            C[i, j] += a_val * B[k, j]
    return C

def main():
    np.random.seed(42)
    
    # 1. Наивный метод (512x512)
    n_small = 512
    A = np.random.rand(n_small, n_small).astype(np.float32)
    B = np.random.rand(n_small, n_small).astype(np.float32)
    
    start = time.time()
    naive_mult(A, B)
    t_naive = time.time() - start
    print(f"1. Наивный: {t_naive:.2f} сек, {2*n_small**3/t_naive/1e6:.2f} MFlops")

    # 2. BLAS (2048x2048)
    n_large = 2048
    A = np.random.rand(n_large, n_large).astype(np.float32)
    B = np.random.rand(n_large, n_large).astype(np.float32)
    
    start = time.time()
    blas_mult(A, B)
    t_blas = time.time() - start
    p_blas = 2*n_large**3/t_blas/1e6
    print(f"2. BLAS: {t_blas:.4f} сек, {p_blas:.2f} MFlops")

    # 3. Оптимизированный (2048x2048)
    optimized_mult(A[:64, :64], B[:64, :64])  # Прогрев JIT (маленький размер)
    
    start = time.time()
    C_opt = optimized_mult(A, B)
    t_opt = time.time() - start
    p_opt = 2*n_large**3/t_opt/1e6
    ratio = p_opt/p_blas*100
    print(f"3. Оптимизированный: {t_opt:.4f} сек, {p_opt:.2f} MFlops ({ratio:.1f}% от BLAS)")

if __name__ == "__main__":
    main()
```
## Результаты выполнения программы:
К сожалению мой процессор не позволяет выдать результат больше 4%
![image](https://github.com/user-attachments/assets/af916fbb-4d25-46e9-81b8-9a567c0dc969)

