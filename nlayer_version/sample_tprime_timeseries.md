# Sample time series: t, t′₂, t′₁(t), t′₁(t′₂) — 15 steps

**Setup:**
- L1 plateau starts at **t = 3**, duration **4** → L1 in plateau for t ∈ {3, 4, 5, 6}.
- L2 plateau starts at **t = 5**, duration **4** → L2 in plateau for t ∈ {5, 6, 7, 8}.
- When not in plateau: t′ = t (current time).

**Conventions:**
- **t** = wall-clock time step.
- **t′₂(t)** = L2’s plateau-start time at step t (constant 5 while L2 is in plateau).
- **t′₁(t)** = L1’s plateau-start time at step t (constant 3 while L1 is in plateau).
- **t′₁ at t′₂** = t′₁ evaluated at the time index t′₂(t), i.e. **t′₁(t′₂(t))** — the value of L1’s t′ when we “sample L1 at t′₂” for the backward pass through L2’s dendrite.

|  t  | t′₂(t) | t′₁(t) | t′₁(t′₂) = t′₁ at t′₂ |
|-----|--------|--------|------------------------|
|  0  |   0    |   0    | t′₁(0) = **0**         |
|  1  |   1    |   1    | t′₁(1) = **1**         |
|  2  |   2    |   2    | t′₁(2) = **2**         |
|  3  |   3    |   3    | t′₁(3) = **3**  (L1 plateau start) |
|  4  |   4    |   3    | t′₁(4) = **3**         |
|  5  |   5    |   3    | t′₁(5) = **3**  (L2 plateau start; sample L1 at 5 → get t′₁=3) |
|  6  |   5    |   3    | t′₁(5) = **3**         |
|  7  |   5    |   7    | t′₁(5) = **3**  (L1 out of plateau; still use value at t′₂=5) |
|  8  |   5    |   8    | t′₁(5) = **3**         |
|  9  |   9    |   9    | t′₁(9) = **9**         |
| 10  |  10    |  10    | t′₁(10) = **10**       |
| 11  |  11    |  11    | t′₁(11) = **11**       |
| 12  |  12    |  12    | t′₁(12) = **12**       |
| 13  |  13    |  13    | t′₁(13) = **13**       |
| 14  |  14    |  14    | t′₁(14) = **14**       |

**Notes:**
- **t′₁(t)** is flat at 3 for t = 3, 4, 5, 6 (L1 plateau).
- **t′₂(t)** is flat at 5 for t = 5, 6, 7, 8 (L2 plateau).
- **t′₁ at t′₂**: For t = 5, 6, 7, 8 we sample L1 at time **t′₂ = 5**, so we always use **t′₁(5) = 3** (L1’s plateau start). So this column is constant 3 over t ∈ {5, 6, 7, 8}, even though t′₁(t) goes 3, 3, 7, 8 — the backward path “freezes” the read at t′₂ = 5 and thus sees t′₁ = 3.
