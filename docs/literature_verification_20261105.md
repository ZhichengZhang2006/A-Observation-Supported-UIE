# 文献与基线来源独立核查（2026-11-05）

## 1. 范围、结论与证据等级

- 截止日期及核查日期：**2026-11-05（Asia/Shanghai）**。对象为[研究任务书](../AutoResearch_A_Observation_Supported_UIE.md)、[文献矩阵](literature_matrix.csv)和只读服务器来源线索；本报告不裁决指标、H1/H2/H3结果或最终创新状态。
- 本轮保留原矩阵45个 `paper_id` 与17列结构。**13 PASS、3 FAIL、29 NOT_AUDITED**；原“43项已查到”不能再表述为43项独立全文核查通过。未核行的旧方法字段仅是历史检索记录，不能进入本轮论证。
- `PASS_20261105`：从正式论文/作者预印本正文核到题名、作者、版本日期及这里明确定位的方法主张；**不意味着结论被独立复现实验验证，也不意味着下载权重认证通过**。
- `FAIL_20261105`：本轮无法取得足够正式全文，不能证实具体方法重合；不表示论文不存在或方法错误。保留历史题录和旧主张作为待核线索，排除当前论证。
- `NOT_AUDITED_20261105`：本轮没有完成正文核验；历史“已查到/尚未核实”写在状态中，不作为证据。基础CVPR2018/2019 PDF请求虽返回200，但正文抽取为空，故也未擅自提升这两行状态。
- 原矩阵 SHA-256：`454e0ad0f883a0a70fd631ae2cf811eb7a91a11d67e83e5c202fdd029ea03444`。下文记录纠错前的关键原句/标签和替换理由；不是悄悄删除失败来源。
- 方法：逐项 `web_fetch` 正式URL；HTML需有方法正文；MCL和SDUIE正式PDF的web_fetch虽200但正文为空，另用内存中 `urllib.request + pypdf` 提取并阅读，**不将空响应、搜索摘要、反爬页、URL存在或仓库可访问计为全文证据**。未安装依赖、未运行GPU、未写服务器文件、未改其余项目文件。

## 2. 优先UIE文献逐项核查

### 2.1 MCL-UIE — PASS（最直接的近期方法重合）

**题名**：Multi-level Constrained Learning for Underwater Image Enhancement。**作者**：Wangtao Bao、Liping Xie、Kanjian Zhang、Huimin Lu。**日期**：arXiv v1，2026-10-29。来源：[题录](https://arxiv.org/abs/2610.39183)、[正式作者PDF](https://arxiv.org/pdf/2610.39183)（13页）。HTML端点提示无HTML，不能拿它作正文。

- PDF pp.4-5：MCT在log-LMS空间以正对角及有界跨通道系数约束颜色变换；固定预测参数下的单调性不等于整个输入依赖网络的全局单调保证。文中使用归一化RGB而非显式逆相机响应后的标定辐亮度。
- pp.5-7，Eq.18-21/31-33：UGWR按RGB通道×三个Haar方向预测残差及 `U=clip[-6,6](head)`，`Q=sigmoid(-U)`，`D_hat=D+Q*DeltaD`。目标 `Q*=exp(-sg(E)/tau)`，`E=|D_hat-D_J|`，tau为各图/色通道归一化误差均值，SmoothL1监督。作者明确这是归一化重建误差的学习权重，**不是“细节系数正确”的概率**。
- p.6，Eq.22-27：RADR从同一瓶颈预测beta/A/d，beta按通道均值归一化；两个路径用同一简化渲染器 `R(X)=X*t+A*(1-t)` 回放预测与成对参考至输入。正文明确参数不是独立测得的光学量或公制深度。训练用参考锚定，推理不运行回放。
- pp.7-11含全参考、无参考比较及校准/回放消融；校准单独作用并非处处有益，完整模型的平均质量提升不等于门控已标定物理信息。表中数值属于论文报告，不是本项目统一评测。

**保守判断**：高频可靠性门控、结构化颜色处理、观测回放均已实质出现。不能靠把误差门控改名“support”建立差异。可检验的增量只能是：独立物理诊断确实测到额外观测信息，且在控制预测风险、退化、平滑和容量后带来机制收益；本报告不宣称这些已成立。

### 2.2 Solomatov / Akkaynak — PASS（理论边界，不是经验基线）

**题名**：Underwater Color Restoration with Vanishing Uncertainty。**作者**：Grigory Solomatov、Derya Akkaynak。**日期**：v1，2026-08-16。[全文](https://arxiv.org/html/2608.15598v1)。

- §I-C/I-E及§III-VI：一般情形颜色反演病态；在无约束函数空间中可出现无界颜色集合。**不能外推为对任意有界RGB先验也无界**。
- 有限区间充分条件使用波长衰减的界，例如 `u1 <= exp(-c(lambda)) <= u2`，并使用每像素距离、至少一个无穷远距离像素和固有辐亮度分段信息。
- 随分辨率提高而不确定性消失的定理还需理想化的分段常数/连通/离散或epsilon分离的辐亮度、连续距离、稠密采样等条件；不要求已知相机光谱灵敏度。

**历史纠错**：原矩阵和[旧新颖性审计](novelty_audit.md)“需要已知水体光学参数”过强，应改为**光学界及特定几何/分段假设**。它不是单张RGB任意场景上可计算的无条件恢复概率，但不能据此认定本文已与其有效区分。

### 2.3 PUIE-Net — PASS（必须保留的机制对照）

**题名**：Uncertainty Inspired Underwater Image Enhancement。**作者**：Zhenqi Fu、Wu Wang、Yue Huang、Xinghao Ding、Kai-Kuang Ma。**日期**：arXiv v1，2022-07-20。[全文](https://arxiv.org/html/2207.09689)、[作者代码](https://github.com/zhenqifu/PUIE-Net)。

§3-4：条件VAE与AdaIN/PAdaIN建模潜在增强外观分布，默认20次Monte Carlo共识。处理成对参考/外观的不确定性，非独立标定的物理可恢复性。UIEBD/RUIE实证可核；本轮没有认证发行权重字节，也未独立复核ECCV正式 proceedings，因此PASS针对作者全文的身份及方法而非所有发行信息。

**边界**：不能把“非物理”当作不用比较的理由；本任务H2明确需要与普通不确定性分离。

### 2.4 BEM — PASS（注意oracle选样）

**题名**：Bayesian Neural Networks for One-to-Many Mapping in Image Enhancement。**作者**：Guoxi Huang、Qirui Yang、Ruirui Lin、Zipeng Qi、David Bull、Nantheera Anantrasirichai。**日期**：arXiv首发2025-01-24，所核v3为2025-11-26；arXiv题录标记AAAI2026，DOI `10.1609/aaai.v40i7.37413`。[全文](https://arxiv.org/html/2501.14265)、[题录](https://arxiv.org/abs/2501.14265)、[代码](https://github.com/BinCVER/BEM)。[AAAI页](https://ojs.aaai.org/index.php/AAAI/article/view/37413)返回Anubis挑战，不是假装读到正文。

§3-4：带自适应先验的变分BNN采样权重生成低维/低通增强代理，再由DNN补细节；既有MC共识，也有排序方案。因此旧“只回答多样性”过窄，其低维外观—细节拆分本身亦为机制近邻。

**关键可比性限制**：§4.1全参考 **BEMRank用参考选择最近样本，作者明确视为上界**。不能把它与普通单输出方法等价纳入性能门槛。R90论文报告MC 23.92/0.913与Rank 25.62/0.940是不同选样协议，均非本项目统一重跑结果。无参考IQA选样还需单独计采样/排序成本。

### 2.5 WWE-UIE — PASS（边缘门控不是概率）

**作者**：Ching-Heng Cheng、Jen-Wei Lee、Chia-Ming Lee、Chih-Chung Hsu。**题名**：WWE-UIE: A Wavelet & White Balance Efficient Network for Underwater Image Enhancement。**日期**：arXiv v1 2025-11-20；WACV March2026 pp.2135–2145。[CVF正式页](https://openaccess.thecvf.com/content/WACV2026/html/Cheng_WWE-UIE_A_Wavelet__White_Balance_Efficient_Network_for_Underwater_WACV_2026_paper.html)、[全文](https://arxiv.org/html/2511.16321)。

§3融合gray-world/log白平衡和Haar增强；§3.3 Eq.8-9 Sobel+sigmoid引导细节。该门控无校准正确概率语义。§4默认8:1:1随机划分和256训练，不能凭712/89/89数量认定本项目测试文件一致。论文有Color-Check7及全参考/无参考评测；这里不导入其分数到统一SOTA表。

### 2.6 SDUIE — PASS（修正错误URL及“无观测模型”）

**完整题名**：SDUIE: Semi-Supervised Diffusion for Underwater Image Enhancement with Quant-Text Dual Control。**作者**：Xiaofeng Cong、Yu-Xin Zhang、Hao Shen、Yeying Jin、Junming Hou、Jie Gui。**日期**：CVPR June2026 pp.37423–37433。[正式页](https://openaccess.thecvf.com/content/CVPR2026/html/Cong_SDUIE_Semi-Supervised_Diffusion_for_Underwater_Image_Enhancement_with_Quant-Text_Dual_CVPR_2026_paper.html)、[全文PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Cong_SDUIE_Semi-Supervised_Diffusion_for_Underwater_Image_Enhancement_with_Quant-Text_Dual_CVPR_2026_paper.pdf)。

- 原矩阵缩略题名对应URL返回404；现已改为正式全题名URL。
- PDF pp.3-5：Quant两LoRA分支分别以合成监督学习增强、以自监督保留水下色调；共享潜在UNet，alpha球面插值控制增强程度。Text以Quant生成样本学习级别语义提示。
- **p.4 Eq.3确有物理合成**：`xs(c)=eta(c)*[y(c)*exp(-beta*d)+L(c)*(1-exp(-beta*d))]`。原“无观测模型”错误；这是简化训练合成模型，不是物理可恢复性标定。
- 1001张陆地图像产生合成配对，结合真实未配对数据；p.5/Table1主要用C60/U45/UCCS/EUVP的UIQM/UCIQE/URANKER，不能直接成为本项目六项PSNR/SSIM验收参考。
- 本轮没完成[代码仓库](https://github.com/Xiaofeng-life/SDUIE)完整性/权重认证；API受限或网页导航截断不支持旧“代码未放全”的确定断言，该断言已撤回。

### 2.7 UADiff — FAIL；保留检索线索，排除方法论证

历史题录：UADiff: A Deep Underwater Image Enhancement Network Using Generative Diffusion Prior and Uncertainty-Aware Learning；历史作者Ou Y./Esmaeilzehi A./Ahmad M. O./Swamy M. N. S.；历史年份2025。**本轮未能凭正式全文核实这些字段及具体重合**。

[DOI](https://doi.org/10.1109/TGRS.2025.3578927)跳至[IEEE11053174](https://ieeexplore.ieee.org/document/11053174/)，HTTP202但无正文。[ResearchGate备选](https://www.researchgate.net/publication/393072057_UADiff_A_Deep_Underwater_Image_Enhancement_Network_using_Generative_Diffusion_Prior_and_Uncertainty-aware_Learning)返回Access restricted。不得把搜索片段当作者方法证据，旧“空间分区依据网络恢复不确定性、与本文语义不同”不能继续支撑最近邻排除。

### 2.8 AquaClarity — FAIL（题录有仓库支持，方法未核）

[官方作者README](https://raw.githubusercontent.com/CXH-Research/AquaClarity/main/README.md)可读，支持题名 AquaClarity: Vision Transformer Enhances Underwater Image via Nonlinear Feature Fusion and Color Fidelity Improvement，作者Xuhang Chen、Zimeng Li、Xiujun Zhang、Yingtie Lei、Yihang Dong、Xiaofeng Zhang、Chi-Man Pun及TCE2025 DOI线索。

但[DOI](https://doi.org/10.1109/TCE.2025.3631911)跳至[IEEE11242133](https://ieeexplore.ieee.org/document/11242133/)返回202空正文；[ResearchGate备选](https://www.researchgate.net/publication/397548691_AquaClarity_Vision_Transformer_Enhances_Underwater_Image_via_Nonlinear_Feature_Fusion_and_Color_Fidelity_Improvement)受限。因此旧“纯结构、无可恢复性”方法排除断言未核，不能用于新颖性论证。官方README简单train/test说明与服务器本地改写版本明显不同；README无可见发布权重链路不能证明当地best.pth官方。

### 2.9 任务书TMM不确定性扩散工作 — FAIL

历史题名 Accelerating Adaptive Diffusion and Uncertainty Modeling for Underwater Image Enhancement，历史DOI [10.1109/TMM.2025.3632678](https://doi.org/10.1109/TMM.2025.3632678)。本轮正式URL抓取TLS断连/后备浏览器失败，没取得作者与方法正文。旧“只建模参考纹理不确定性”保留为历史线索，不得用于区别论证。

## 3. 逆问题与信息量基础的已核正文

以下均于2026-11-05读到正式作者全文。PASS仅覆盖定位主张，不能把跨领域定理直接移植到未知水参数的单RGB UIE。

| 文献、作者、版本日期与来源 | 可核正文 | 必要纠正/限制 |
| --- | --- | --- |
| On Hallucinations in Tomographic Image Reconstruction；Sayantan Bhadra、Varun A. Kelkar、Frank J. Brooks、Mark A. Anastasio；v3 2021-09-27；[全文](https://arxiv.org/html/2012.00646) | §II-B/III：截断SVD的generalized measurement/null分解及hallucination maps | 原“线性高斯”过窄，核心为线性算子与稳定性截断；一般参考误差不等于幻觉，null部分判定可能依赖真值。 |
| Spotting Hallucinations in Inverse Problems with Data-Driven Priors；Matt L. Sampson、Peter Melchior；v1 2023-06-23；[全文](https://arxiv.org/html/2306.13272) | §2 Eq.5：`DeltaF=Diag(F_logP)-Diag(F_logL)`；Hutchinson/HVP；§3星系修复/去噪 | 原“ratio”错误，实际difference。作者使用log曲率近似；不是任意耦合问题的完整可辨识性证明。已有信息来源分离图像先例。 |
| Inequalities and Approximations for Fisher Information in the Presence of Nuisance Parameters；Eric Clarkson；v1 2019-02-12；[全文](https://arxiv.org/html/1902.04607) | §2：边缘化nuisance后信息不大于条件信息的先验平均；§5窄先验近似 | 平均界不是每个nuisance取值逐点单调；marginalization不等于profile/Schur；不认证plug-in参数为真值。 |
| Local Information Operators for Spatial Identifiability in Distributed-Parameter Inverse Problems in Computational Mechanics；Tammam Bakeer；v1 2026-05-27；[全文](https://arxiv.org/html/2605.28601) | §3局部噪声加权Gramian/Fisher/Gauss-Newton联系；§3.4对角、坐标与度量限定 | 局部线性与参数无关Gaussian协方差等假设；对角量不等于完整空间可辨识性。仅认证为预印本，未证明UIE增量。 |
| Measurement Geometry and Design for Trustworthy Generative Inverse Problems；Pengfei Jin、Na Li、Quanzheng Li；v1 2026-06-01；[全文](https://arxiv.org/html/2606.02309) | §3 `alpha(A;x*)=sigma_min(A U*)`；§4局部稳定性条件；§6 MNIST/CT/FastMRI | 不只是算子设计，也含观测几何与恢复可靠性的实证。需流形/局部正则性/先验与局部解假设，不是无条件Fisher概率。 |
| Looks Too Good To Be True: An Information-Theoretic Analysis of Hallucinations in Generative Restoration Models；Regev Cohen、Idan Kligvasser、Ehud Rivlin、Daniel Freedman；v3 2024-10-25；[全文](https://arxiv.org/html/2405.16475) | §3-4 conditional entropy-power `N(X|Y)`及完美条件感知的至少2N界；SR/inpainting实证 | 不能将其表述为一般像素MSE、局部Fisher或任意恢复器下的2倍界。旧“无法与恢复器耦合”绝对否定撤回。 |
| Principal Uncertainty Quantification With Spatial Correlation for Image Restoration Problems；Omer Belhasin、Yaniv Romano、Daniel Freedman、Ehud Rivlin、Michael Elad；v3 2024-01-20；[全文](https://arxiv.org/html/2305.10124) | §III Eq.2加权miscoverage；§IV后验样本SVD/PCA与held-out Learn-then-Test，E-PUQ/DA-PUQ | 原作者H.应为Omer。已有主方向分解，不能说无component。保证需校准条件，不是无条件全图同时包含真值；本轮未另核TPAMI DOI。 |

## 4. 最多三项最近实证近邻：保守工作清单

这里“近邻”按对本任务**测量—决策—经验验证链条**的相关度，而不是声称对所有2026论文完成系统排序；不把理论Solomatov强行归为实验方法，也不把FAIL的UADiff算作已核近邻。

1. **MCL-UIE（2026-10）**：同UIE且同时有颜色约束、方向细节门控、参考误差校准、共享参数回放，是最直接结构重合。拟议方法要证明门控量的独立物理含义与额外机制收益；非“换一个名字”即可过关。
2. **BEM（AAAI2026；作者预印本2025）**：同UIE的一对多不确定性与低维外观/细节拆分，必须比较共识、普通选样和成本；oracle Rank只能诊断。它不能代替PUIE-Net基础对照，PUIE仍必须保留。
3. **Jin等 Measurement Geometry（2026-06）**：跨域但已将测量—先验切向兼容度与生成反演可靠性作实证联系。它否定“观测支持预测失败从未被做过”的宽泛说法。真正可能的增量需限于经验证的未知水参数、颜色/频段诊断及实际UIE决策，而非沿用测量几何概念本身。

**现有证据只支持“存在可具体检验的差异”，不支持“已证实新颖”、‘首次’或‘所有先例都没有分量测量’。** 尚未核实的29项及3项FAIL意味着本轮不是完整穷尽检索。

## 5. 只读SSH：服务器模型与权重来源风险

核查目标 `school-gpu2`；只读取[服务器目录](</home/user>)下九个模型文件夹README、Git状态和有限深度权重路径。没有载入模型、反序列化权重或运行GPU。

对每个目录运行如下只读检查：

```sh
git -C /home/user/<repo> rev-parse --is-inside-work-tree
git -C /home/user/<repo> remote -v
```

**九个目录均返回** `fatal: not a git repository (or any parent up to mount point /home)` 以及 `Stopping at filesystem boundary (GIT_DISCOVERY_ACROSS_FILESYSTEM not set).` 因此不是有可检查remote/commit的Git工作树；可能是归档副本。没有remote本身不证明权重由用户训练，但也绝不认证官方来源。

| 仓库/只读证据 | 所见权重或说明 | 来源风险判断 |
| --- | --- | --- |
| [Phaseformer](</home/user/Phaseformer-main>) README自称WACV2025官方 | README提及`checkpoints`；本地例为[epoch266](</home/user/Phaseformer-main/checkpoint/UFO/netG_model_epoch_266.pth>)等大量逐epoch快照 | 目录名/官方README不能证明所选epoch是发行权重；尚无下载链、发布commit、字节digest。 |
| [AquaClarity](</home/user/AquaClarity-main>) README改为本地数据根及分数据集训练CLI | [UIEB best](</home/user/AquaClarity-main/checkpoints/UIEB/best.pth>)及LSUI/UFO等best；官方在线README与本地明显不同 | 有强本地改造/训练流程证据；不能断言每个文件必为用户训练，但不能标官方发布。 |
| [FDCE-Net](</home/user/FDCE-Net-main>) README明确适配UIEB712/89/89、LSUI3423/428/428、UFO1200/300/120并给训练CLI | [UIEB best](</home/user/FDCE-Net-main/training_data/UIEB/checkpoints/best.pth>)与last；说明UFO GT640×480下采样至输入320×240做增强 | 强本地训练/协议适配证据，不是已认证官方权重。数量不能证明标准测试文件。 |
| [CDF-UIE](</home/user/CDF-UIE-main>) README存在本地数据训练示例 | [timestamp/epoch92](</home/user/CDF-UIE-main/ckpts/UIEB-0730-1508/UHDFour-epoch92-13.42676.pt>)等带epoch/分数文件 | 时间戳与模型名是训练产物线索，不是官方发布链。 |
| [SS-UIE](</home/user/SS-UIE-main>) README既有原LSUI3879/400及外部预训练链接，又附本地分数据集训练修改 | [UFO epoch609](</home/user/SS-UIE-main/saved_models/UFO/SS_UIE_epoch_609.pth>)等 | 可见公开链接不等于当前所选文件的同一性；需核对发行字节。 |
| [CCL-Net](</home/user/CCL-Net-main>) README保留原说明 | [LSUI best](</home/user/CCL-Net-main/checkpoints/LSUI_CC/best_net_G_CC.pth>)、100/latest等 | 仅凭best/latest命名无法区分用户训练和公开权重。 |
| [SFGNet](</home/user/SFGNet-main>) README称预训练模型未来上传 | [UFO epoch41](</home/user/SFGNet-main/models/UFO/net_epoch41.pth>)等 | README与本地epoch存在不能组合成已下载官方发布的证据；来源未认证。 |
| [WWE-UIE](</home/user/WWE-UIE-main>) README有[UIEB公开链接](https://drive.google.com/file/d/1c80kkDbgpLD1MtqVLql1Y1NQVD7IzxiO/view?usp=sharing) | 深度4范围只见[URanker](</home/user/WWE-UIE-main/utils/uranker/URanker_ckpt.pth>)与[CIDNet低光辅助权重](</home/user/WWE-UIE-main/utils/CIDNet_weight_LOLv2_bestSSIM.pth>) | 它们不是WWE增强主权重；有限深度未见不等于全盘不存在。不能把可见辅助pth作为模型已可复现。 |
| [MambaCue](</home/user/MambaCue>) README为用户variant07并列SpatialColorCorrection/HighFreqDetailInjector等 | 本轮有限深度所见辅助指标权重，未建立增强权重官方发行链 | 用户变体/保存输出不能自行变成“官方SOTA”。旧结果拼图重计也不是加载官方权重统一重跑。 |

**分类建议**（交父代理处理，本轮不修改SOTA表）：

- 将“架构代码来自论文仓库”和“该checkpoint为官方发布”分开。当前最多称**本地可用checkpoint重跑、发行来源未认证**；不能直接填 `official_weights_rerun`。
- 如果训练记录、配置、固定选checkpoint规则和数据manifest可追溯，可另列“本地训练/复训”轨道；仅有适配README不足以宣称协议匹配复训。
- 认证官方权重需实际选择文件的路径、SHA-256、发布URL、版本/commit或作者提供digest与下载链。无来源时保留结果但限制比较范围，不伪造来源字段。
- 服务器README中的本地划分与原论文标准划分可能不同；此次只核来源，不重复父代理的数据/指标审计，也不因数值接近论文就倒推来源正确。

## 6. 历史纠错台账与交接

| 原记录中的主张/标签 | 本轮处理 |
| --- | --- |
| 43行“已查到”被视为核查通过 | 改为13 PASS、3 FAIL、29 NOT_AUDITED；每行保留历史状态。 |
| Solomatov“需要已知水体光学参数与场景几何” | 改为光学衰减界及定理特定几何/分段条件；精确已知光学量并非所核前提。 |
| Sampson“likelihood versus prior information ratio” | Eq.5实际为difference；矩阵改正。 |
| PUQ作者“Belhasin H.”、保证包含全部真值、不给分量级分解 | Omer；加权miscoverage及条件化校准；已有主方向分解。 |
| SDUIE短题名404 URL、“无观测模型”、仓库“未放全” | 正式全题名URL；Eq.3物理合成；仓库完整性未核，不沿用确定否定。 |
| Bhadra仅“线性高斯” | 缩小到已核的线性算子/截断稳定性定义，不硬添Gaussian限制。 |
| BEM只回答多样性；Rank当普通成绩 | 低维代理+DNN细节也是机制重合；全参考Rank为oracle上界。 |
| UADiff/Aqua/TMM“已查到”并据此排除方法重合 | 标FAIL且禁止用于排除论证；历史题录与方法字段只供后续定位。 |
| 本地README称官方、存在best.pth、数值接近即可认定官方权重 | 九目录均非Git树；checkpoint发行链未认证。保留本地结果但需正确来源标签。 |

[旧新颖性审计](novelty_audit.md)不在本轮写权限中，**未修改**；父代理应同步以上纠错，尤其删除未核UADiff/TMM的确定排除及“无分量先例”等过强判断。本报告不改评分器、不调整研究验收阈值，不替父代理判H2/H3胜负。

剩余缺口仅列三项：
1. 取得UADiff/Aqua/TMM正式作者全文后再恢复方法比较；29项NOT_AUDITED若用于论文也需逐项核。
2. 为实际评测checkpoint建立发布或本地训练来源链及固定选择规则。
3. 由父代理在修正来源标签和方法边界后更新新颖性审计；不能把本轮文献PASS当作新方法创新PASS。
