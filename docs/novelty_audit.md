# Novelty Audit — Observation-Supported Underwater Image Enhancement

配套清单：`docs/literature_matrix.csv`（45 条）。检索面：arXiv API、Crossref、OpenAlex、OpenReview、GitHub Contents API、CVF 开放获取。检索日 2026-11-04/05。

## 一、最近三年最接近的三篇

**Solomatov & Akkaynak, arXiv:2608.15598（2026-08）｜已查到。** 纯解析，无网络、无监督、无数据集评测。它先把水下颜色恢复判为一般情形下的病态问题，指出需附加约束才能把解收进**有限不确定性区间**，再区分"理论上够用的约束"与"真实数据满足的约束"，最后给出一组理想条件使不确定性有界且随相机分辨率趋零。差别在前提（**2026-11-05 按原文修正**）：它不要求精确已知水体光学参数，而要求衰减系数**界**、距离、无穷远像素/inherent-radiance 分割与理想可分性——是存在性充分条件，不是单图可算的量；我们要在单张 RGB、光学参数未知下算出分量支持度，并做恢复器耦合实验。

**Bao 等, MCL-UIE, arXiv:2610.39183（2026-10）｜已查到，代码公开、权重未核实。** 设定最接近。动机句即：成对参考监督无法规定颜色映射、局部细节修正强度、以及与退化观测的一致性。方法三件：log-LMS 空间的单调色彩变换约束全局映射；**不确定性门控小波重建**在 RGB×Haar 细节带上用方向置信门 `Q = sigmoid(-U)` 调制高频残差，`U` 对归一化重建误差做校准（目标 `exp(-sg(E)/tau)`）；训练期用参考锚定的退化重放。作者自述该门是"归一化重建误差的权重"，不是"系数正确的概率"——这正是我们必须切开的口子，但它比"普通 confidence 图"更接近，不能轻描淡写。评价为 16 项全参考指标比较（称 13 项第一）加消融。关键区别：其 gate 是对重建误差的回归，不由成像模型与噪声协方差导出。

**Ou 等, UADiff, IEEE TGRS 2025, DOI 10.1109/TGRS.2025.3578927｜已查到。** 主干为扩散先验，两个想法是 uncertainty-aware learning（按各区域恢复不确定性决定处理方式）与按主导色调的特征重校准。评价为标准 UIE benchmark 的 PSNR/SSIM 加感知指标。它证明"不确定性图→空间分区→差异化处理"这条链在 UIE 里已跑通。

同层：PUIE-Net（ECCV 2022）与 BEM（AAAI 2026）做的是数据/分布不确定性；Zeng 等（TMM 2025）建模参考纹理不确定性且同样做低频/高频分离。三者说明"不确定性＋频段"不是空位。

**独立核查后的更正（2026-11-05，见 `docs/literature_verification_20261105.md`）**：SDUIE（CVPR）第 4 页 Eq.3 确实含物理成像合成，原矩阵"无观测模型"标注已改；Sampson & Melchior Eq.5 是对角信息的**差** $\Delta F$ 而非比值；PUQ 作者为 Omer 且已做**主方向分解**，不能声称前人未做分量。UADiff/Aqua/TMM 三篇本轮未取到正式全文（FAIL），因此它们**不能**再承担新颖性论证，只能作为检索线索。45 条里 13 条 PASS、3 条 FAIL、29 条未审——**文献 PASS ≠ 本方法创新 PASS**。

## 二、新测量对象，还是换名

**对象一不是新代数。** 已查到 Sampson & Melchior（arXiv:2306.13272）已把似然与先验的 Fisher 对角分开算，用信息比标记先验主导区；Clarkson（arXiv:1902.04607）已给出 nuisance 参数下 FIM 的不等式与近似；Bakeer（arXiv:2605.28601）已明写 nuisance 通过 covariance inflation、Schur 补信息损失、先验预条件模态修改同一几何。故 $q_v$ 不能作为独立创新宣称，只能是"把上述工具落到 UIE 分量上"的一部分，且不作任何优先性声明。

**对象二作为测量目标不同。** Floros 等（arXiv:2505.12375）形式化了"对测量的一致性"并要求结果落入一致集，这是**可行性**；我们要量化该集合在某颜色/频率方向上的**宽度**，是**可分辨性**。但 Korolev & Lellmann（SIAM J. Imaging Sci. 11(1):197-218, 2018）与凸集投影传统早已用 order intervals 描述模型误差下的可行集，Belhasin 等（TPAMI 2024）已占用 "ambiguity region"。故这两说法不是我们的；可主张的仅是"按分量拆开、把 ψ profiled 掉后仍可算"。

**结论：不是换名，但极近，必须靠一条证据兜住**——该量**脱离恢复器**仍有预测力（与独立测得的分量误差单调对应），而 confidence map 只在绑定训练过的网络时有意义。做不到，就应承认同类。

## 三、会推翻区分度的实验判据

任一条成立即判定不成立：

1. **平凡基线吞掉收益**：以"仅由噪声 σ 与透射率 t 构成的标量阈值"或局部方差图作同一门控输入，同骨干同预算、匹配参数量后，三数据集 ΔPSNR ≤ 0.1 dB 且 ΔSSIM ≤ 0.001。
2. **只认合成标签**：合成轨道 Spearman ρ 的 95% CI 下界 > 0.3，但 C60/U45/RUIE 真实轨道 ρ 的 CI 覆盖 0；或换模拟器（改 PSF/噪声模型）后 CI 覆盖 0。
3. **退化为低通**：错误边缘率下降时真实边缘召回同幅下降，且 work point 在 recall–false-edge ROC 上被固定强度低通/双边滤波支配。
4. **成本归一后消失**：计入 Fisher/Schur 开销后同 wall-clock 预算重训对照，ΔPSNR ≤ 0.1 dB；或收益仅在"支持度作额外监督"的骨干上出现，换骨干后 ρ 符号翻转或 CI 覆盖 0。
5. **与现成不确定性图不可分**：同骨干把门控输入换为 MC-dropout 方差、深度集成方差、我们的支持度，两两 ΔPSNR ≤ 0.1 dB，且 risk–coverage 在 90% 覆盖处宽度差 ≤ 5%。此条最致命，直接回答"是否换名"。

否证纪律：若模糊半径对任意合理 $(C,\Psi,\tau)$ 都随 τ 线性缩放而无稳定结构，则它只是 τ 的重新参数化，须作负结果报告。

## 四、我们没有主张什么

- 不主张单张 RGB 能无条件判断真实场景可恢复性；一切限定在给定观测模型、噪声水平、允许场景集合与误差容限内。
- 不主张 $F(I)$ 是完整可行域或严格信息极限，只称**操作性歧义估计**（C、Ψ、τ 均来自有限样本与优化候选）。
- 不主张 Fisher/Schur 代数、nuisance 处理、各向信息损失、ambiguity region 为原创（Sampson & Melchior 2023、Clarkson 2019、Bakeer 2026、Belhasin 等 2024、Korolev & Lellmann 2018）。
- 不主张"存在恢复上限"是发现（Treibitz & Schechner ICCP 2009 的噪声致截止频率；Cohen 等 NeurIPS 2024 的固有不确定性下界；Kang 等 2022 的信息边界）。
- 不主张"不确定性门控＋高频增强"是新的（MCL-UIE 2026、UAR-Net 2026、DACG-IR 2026、UADiff 2025、TMM 2025）。
- 不主张"预测可恢复性"是新任务（IRP, CVPR 2022）；不主张 SOTA，只主张匹配质量与成本下该对象含不可替代信息。
- 空白陈述一律为"未查到"并附检索式：arXiv 全库 `abs:"underwater" AND abs:"image formation model" AND abs:"uncertainty"` 命中 0；`abs:"Fisher information" AND abs:underwater` 的 10 条全为定位/通信 CRB；`ti:"underwater" AND (abs:"confidence" OR abs:"uncertainty map") AND abs:"enhancement"` 仅 2 条（MCL-UIE 与一篇 AUV 跟踪）。仅说明该库该式下未查到，非不存在性证明。
