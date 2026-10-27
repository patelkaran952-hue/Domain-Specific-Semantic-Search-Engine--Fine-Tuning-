# Held-out test failure cases: ft_hardneg_dense

> DESCRIPTIVE ONLY. The model is frozen; nothing may be tuned after reading these test cases.

> Retrieved documents that are not judged positives may still be relevant (qrels are sparse).

**Best-positive rank distribution — ft_hardneg_dense:** {'rank 1': 181, 'rank 2-10': 71, 'rank 11-100': 31, 'not in top 100': 17}

**Best-positive rank distribution — baseline:** {'rank 1': 151, 'rank 2-10': 87, 'rank 11-100': 40, 'not in top 100': 22}

## No judged positive in the top 10 (48 of 300 queries; showing 8)

### Query 1
**Claim:** 0-dimensional biomaterials show inductive properties.

**Judged positive(s):** [31715818] New opportunities: the use of nanotechnologies to manipulate and track stem cells.
- best-positive rank: ft_hardneg_dense=34, baseline=5  |  nDCG@10: ft_hardneg_dense=0.000, baseline=0.387
- ft_hardneg_dense #1 (cos 0.314): [29638116] Complex Tissue and Disease Modeling using hiPSCs.
- ft_hardneg_dense #2 (cos 0.311): [4346436] Nonlinear Elasticity in Biological Gels
- ft_hardneg_dense #3 (cos 0.288): [10786948] An efficient nonviral method to generate integration-free human-induced pluripotent stem cells from cord blood

### Query 1024
**Claim:** Recurrent mutations occur frequently within CTCF anchor sites adjacent to oncogenes.

**Judged positive(s):** [5373138] 3D Chromosome Regulatory Landscape of Human Pluripotent Cells.
- best-positive rank: ft_hardneg_dense=13, baseline=41  |  nDCG@10: ft_hardneg_dense=0.000, baseline=0.000
- ft_hardneg_dense #1 (cos 0.518): [8494570] Characterization of constitutive CTCF/cohesin loci: a possible role in establishing topological domains in mam
- ft_hardneg_dense #2 (cos 0.485): [11117679] Trans-ancestry mutational landscape of hepatocellular carcinoma genomes
- ft_hardneg_dense #3 (cos 0.482): [4926049] TRF2 Recruits RTEL1 to Telomeres in S Phase to Promote T-Loop Unwinding

### Query 1110
**Claim:** Suboptimal nutrition is not predictive of chronic disease

**Judged positive(s):** [13770184] Global, regional, and national comparative risk assessment of 79 behavioural, environmental and occupational, 
- best-positive rank: ft_hardneg_dense=28, baseline=None  |  nDCG@10: ft_hardneg_dense=0.000, baseline=0.000
- ft_hardneg_dense #1 (cos 0.501): [4345757] Obesity as a medical problem.
- ft_hardneg_dense #2 (cos 0.487): [17324544] Thirty-One Novel Biomarkers as Predictors for Clinically Incident Diabetes
- ft_hardneg_dense #3 (cos 0.486): [24273592] Dietary quality among men and women in 187 countries in 1990 and 2010: a systematic assessment

### Query 1199
**Claim:** The benefits of colchicine were achieved with effective widespread use of secondary prevention strategies such as high-dose statins.

**Judged positive(s):** [16760369] Comparative determinants of 4-year cardiovascular event rates in stable outpatients at risk of or with atherot
- best-positive rank: ft_hardneg_dense=None, baseline=None  |  nDCG@10: ft_hardneg_dense=0.000, baseline=0.000
- ft_hardneg_dense #1 (cos 0.454): [7454794] Statins in the primary prevention of cardiovascular disease
- ft_hardneg_dense #2 (cos 0.422): [17925632] Efficacy of monthly tafenoquine for prophylaxis of Plasmodium vivax and multidrug-resistant P. falciparum mala
- ft_hardneg_dense #3 (cos 0.420): [14260013] Prevention of Rectal SHIV Transmission in Macaques by Daily or Intermittent Prophylaxis with Emtricitabine and

### Query 1200
**Claim:** The binding orientation of the ML-SA1 activator at hTRPML2 is different from the binding orientation of the ML-SA1 activator at hTRPML1.

**Judged positive(s):** [3441524] Human TRPML1 channel structures in open and closed conformations
- best-positive rank: ft_hardneg_dense=None, baseline=None  |  nDCG@10: ft_hardneg_dense=0.000, baseline=0.000
- ft_hardneg_dense #1 (cos 0.365): [16270577] Nonclassical binding of formylated peptide in crystal structure of the MHC class lb molecule H2-M3
- ft_hardneg_dense #2 (cos 0.332): [12956194] The Extracellular Surface of the GLP-1 Receptor Is a Molecular Trigger for Biased Agonism
- ft_hardneg_dense #3 (cos 0.327): [493346] Evidence that the Ipl1-Sli15 (Aurora Kinase-INCENP) Complex Promotes Chromosome Bi-orientation by Altering Kin

### Query 1226
**Claim:** The loss of the TET protein functions may have dire biological consequences, such as myeloid cancers.

**Judged positive(s):** [13777138] Tet1 and Tet2 regulate 5-hydroxymethylcytosine production and cell lineage specification in mouse embryonic st
- best-positive rank: ft_hardneg_dense=None, baseline=None  |  nDCG@10: ft_hardneg_dense=0.000, baseline=0.000
- ft_hardneg_dense #1 (cos 0.480): [2679511] The BLM helicase contributes to telomere maintenance through processing of late-replicating intermediate struc
- ft_hardneg_dense #2 (cos 0.470): [1982286] Reverse engineering of TLX oncogenic transcriptional networks identifies RUNX1 as tumor suppressor in T-ALL
- ft_hardneg_dense #3 (cos 0.465): [13770184] Global, regional, and national comparative risk assessment of 79 behavioural, environmental and occupational, 

### Query 1245
**Claim:** The one-child policy has been successful in lowering population growth.

**Judged positive(s):** [7662395] Perinatal mortality in rural China: retrospective cohort study.
- best-positive rank: ft_hardneg_dense=13, baseline=9  |  nDCG@10: ft_hardneg_dense=0.000, baseline=0.301
- ft_hardneg_dense #1 (cos 0.478): [11748341] Evidence-based interventions for improvement of maternal and child nutrition: what can be done and at what cos
- ft_hardneg_dense #2 (cos 0.413): [4791384] Neonatal Mortality Levels for 193 Countries in 2009 with Trends since 1990: A Systematic Analysis of Progress,
- ft_hardneg_dense #3 (cos 0.402): [40500723] Immigration cohorts and residential overcrowding in southern California

### Query 1272
**Claim:** The single flash-evoked ERG b-wave is generated by activity of ON-bipolar cells.

**Judged positive(s):** [17081238] Specific deficit of the ON response in visual transmission by targeted disruption of the mGIuR6 gene
- best-positive rank: ft_hardneg_dense=86, baseline=28  |  nDCG@10: ft_hardneg_dense=0.000, baseline=0.000
- ft_hardneg_dense #1 (cos 0.409): [15685921] Differential effects of charybdotoxin on the activity of retinal ganglion cells in the dark- and light-adapted
- ft_hardneg_dense #2 (cos 0.407): [24760136] A naturally occurring mouse model of X-linked congenital stationary night blindness.
- ft_hardneg_dense #3 (cos 0.389): [13757347] Involvement of cortical fast-spiking parvalbumin-positive basket cells in epilepsy.

## Largest regressions vs baseline (nDCG@10)

### Query 1278
**Claim:** The treatment of cancer patients with co-IR blockade does not cause any adverse autoimmune events.

**Judged positive(s):** [11335781] Is autoimmunity the Achilles' heel of cancer immunotherapy?
- best-positive rank: ft_hardneg_dense=18, baseline=1  |  nDCG@10: ft_hardneg_dense=0.000, baseline=1.000
- ft_hardneg_dense #1 (cos 0.422): [9767444] Epigenetic therapy activates type I interferon signaling in murine ovarian cancer to reduce immunosuppression 
- ft_hardneg_dense #2 (cos 0.414): [39443128] Interferon alpha and zidovudine therapy in adult T-cell leukaemia lymphoma: response and outcome in 15 patient
- ft_hardneg_dense #3 (cos 0.408): [3899896] Elevated red blood cell distribution width contributes to poor prognosis in patients undergoing resection for 

### Query 1279
**Claim:** The treatment of cancer patients with co-IR blockade precipitates adverse autoimmune events.

**Judged positive(s):** [11335781] Is autoimmunity the Achilles' heel of cancer immunotherapy?
- best-positive rank: ft_hardneg_dense=11, baseline=1  |  nDCG@10: ft_hardneg_dense=0.000, baseline=1.000
- ft_hardneg_dense #1 (cos 0.447): [6334188] History of chronic comorbidity and risk of chemotherapy-induced febrile neutropenia in cancer patients not rec
- ft_hardneg_dense #2 (cos 0.440): [25589047] Treatment-related mortality with bevacizumab in cancer patients: a meta-analysis.
- ft_hardneg_dense #3 (cos 0.421): [3471191] Association of Pembrolizumab With Tumor Response and Survival Among Patients With Advanced Melanoma.

### Query 183
**Claim:** Bone marrow cells contribute to adult macrophage compartments.

**Judged positive(s):** [12827098] Tissue-resident macrophages self-maintain locally throughout adult life with minimal contribution from circula
- best-positive rank: ft_hardneg_dense=4, baseline=1  |  nDCG@10: ft_hardneg_dense=0.431, baseline=1.000
- ft_hardneg_dense #1 (cos 0.552): [7521113] Fate mapping reveals origins and dynamics of monocytes and tissue macrophages under homeostasis.
- ft_hardneg_dense #2 (cos 0.496): [3495456] Developmental Analysis of Bone Marrow Neutrophils Reveals Populations Specialized in Expansion, Trafficking, a
- ft_hardneg_dense #3 (cos 0.488): [22406695] Origin and functions of tissue macrophages.

### Query 439
**Claim:** Fz/PCP-dependent Pk localizes to the anterior membrane of neuroectoderm cells during zebrafish neuralation

**Judged positive(s):** [4423559] Planar cell polarity signalling couples cell division and morphogenesis during neurulation
- best-positive rank: ft_hardneg_dense=3, baseline=1  |  nDCG@10: ft_hardneg_dense=0.500, baseline=1.000
- ft_hardneg_dense #1 (cos 0.484): [9629682] A Whole Brain Staining, Embedding, and Clearing Pipeline for Adult Zebrafish to Visualize Cell Proliferation a
- ft_hardneg_dense #2 (cos 0.470): [22178316] Disrupted-in-schizophrenia 1 and neuregulin 1 are required for the specification of oligodendrocytes and neuro
- ft_hardneg_dense #3 (cos 0.461): [4423559] Planar cell polarity signalling couples cell division and morphogenesis during neurulation

### Query 728
**Claim:** Ly6C hi monocytes have a lower inflammatory capacity than Ly6C lo monocytes.

**Judged positive(s):** [36444198] Subpopulations of mouse blood monocytes differ in maturation stage and inflammatory response.; [7521113] Fate mapping reveals origins and dynamics of monocytes and tissue macrophages under homeostasis.
- best-positive rank: ft_hardneg_dense=7, baseline=2  |  nDCG@10: ft_hardneg_dense=0.204, baseline=0.651
- ft_hardneg_dense #1 (cos 0.604): [22406695] Origin and functions of tissue macrophages.
- ft_hardneg_dense #2 (cos 0.530): [22621251] Ly6Chi monocytes direct alternatively activated profibrotic macrophage regulation of lung fibrosis.
- ft_hardneg_dense #3 (cos 0.515): [17934603] CCR2 and CXCR4 regulate peripheral blood monocyte pharmacodynamics and link to efficacy in experimental autoim

### Query 727
**Claim:** Ly6C hi monocytes have a lower inflammatory capacity compared to their Ly6C lo counterparts.

**Judged positive(s):** [7521113] Fate mapping reveals origins and dynamics of monocytes and tissue macrophages under homeostasis.
- best-positive rank: ft_hardneg_dense=14, baseline=4  |  nDCG@10: ft_hardneg_dense=0.000, baseline=0.431
- ft_hardneg_dense #1 (cos 0.595): [22406695] Origin and functions of tissue macrophages.
- ft_hardneg_dense #2 (cos 0.522): [17934603] CCR2 and CXCR4 regulate peripheral blood monocyte pharmacodynamics and link to efficacy in experimental autoim
- ft_hardneg_dense #3 (cos 0.520): [22483580] Disturbed sleep in bipolar disorder is related to an elevation of IL-6 in peripheral monocytes.

### Query 1
**Claim:** 0-dimensional biomaterials show inductive properties.

**Judged positive(s):** [31715818] New opportunities: the use of nanotechnologies to manipulate and track stem cells.
- best-positive rank: ft_hardneg_dense=34, baseline=5  |  nDCG@10: ft_hardneg_dense=0.000, baseline=0.387
- ft_hardneg_dense #1 (cos 0.314): [29638116] Complex Tissue and Disease Modeling using hiPSCs.
- ft_hardneg_dense #2 (cos 0.311): [4346436] Nonlinear Elasticity in Biological Gels
- ft_hardneg_dense #3 (cos 0.288): [10786948] An efficient nonviral method to generate integration-free human-induced pluripotent stem cells from cord blood

### Query 1282
**Claim:** Therapeutic use of the drug Dapsone to treat pyoderma gangrenous is based on anecdotal evidence.

**Judged positive(s):** [23649163] Clinical features and treatment of peristomal pyoderma gangrenosum.
- best-positive rank: ft_hardneg_dense=2, baseline=1  |  nDCG@10: ft_hardneg_dense=0.631, baseline=1.000
- ft_hardneg_dense #1 (cos 0.387): [7074001] Evidence into practice. Application of psychological models of change in evidence-based implementation.
- ft_hardneg_dense #2 (cos 0.374): [23649163] Clinical features and treatment of peristomal pyoderma gangrenosum.
- ft_hardneg_dense #3 (cos 0.361): [8042158] Systematic reviews from astronomy to zoology: myths and misconceptions.

## Largest improvements vs baseline (nDCG@10)

### Query 690
**Claim:** Less than 10% of the gabonese children with Schimmelpenning-Feuerstein-Mims syndrome (SFM) had a plasma lactate of more than 5mmol/L.

**Judged positive(s):** [18750453] Assessment of Volume Depletion in Children with Malaria
- best-positive rank: ft_hardneg_dense=1, baseline=None  |  nDCG@10: ft_hardneg_dense=1.000, baseline=0.000
- ft_hardneg_dense #1 (cos 0.367): [18750453] Assessment of Volume Depletion in Children with Malaria
- ft_hardneg_dense #2 (cos 0.362): [3770750] Sugar-sweetened beverages and weight gain in children and adults: a systematic review and meta-analysis.
- ft_hardneg_dense #3 (cos 0.351): [36124058] Update on technical issues concerning complementary feeding of young children in developing countries and impl

### Query 644
**Claim:** Insulin increases risk of severe kidney failure.

**Judged positive(s):** [13619127] Diabetes treatments and risk of amputation, blindness, severe kidney failure, hyperglycaemia, and hypoglycaemi
- best-positive rank: ft_hardneg_dense=1, baseline=28  |  nDCG@10: ft_hardneg_dense=1.000, baseline=0.000
- ft_hardneg_dense #1 (cos 0.490): [13619127] Diabetes treatments and risk of amputation, blindness, severe kidney failure, hyperglycaemia, and hypoglycaemi
- ft_hardneg_dense #2 (cos 0.470): [24921368] Impaired awareness of hypoglycaemia: a review.
- ft_hardneg_dense #3 (cos 0.469): [43483151] QT and QTc dispersion are accurate predictors of cardiac death in newly diagnosed non-insulin dependent diabet

### Query 587
**Claim:** In transgenic mice harboring green florescent protein under the control of the Sox2 promoter, less than ten percent of the cells with green florescent colocalize with cell proliferation markers.

**Judged positive(s):** [16999023] Cell Stem Cell Article In Vivo Fate Analysis Reveals the Multipotent and Self-Renewal Capacities of Sox2 + Neu
- best-positive rank: ft_hardneg_dense=1, baseline=36  |  nDCG@10: ft_hardneg_dense=1.000, baseline=0.000
- ft_hardneg_dense #1 (cos 0.501): [16999023] Cell Stem Cell Article In Vivo Fate Analysis Reveals the Multipotent and Self-Renewal Capacities of Sox2 + Neu
- ft_hardneg_dense #2 (cos 0.498): [23208167] Catalytic-Independent Functions of PARP-1 Determine Sox2 Pioneer Activity at Intractable Genomic Loci.
- ft_hardneg_dense #3 (cos 0.463): [471735] Two-stage control of an oxidative stress regulon: the Escherichia coli SoxR protein triggers redox-inducible e

### Query 415
**Claim:** Female carriers of the Apolipoprotein E4 (APOE4) allele have increased risk for dementia.

**Judged positive(s):** [6309659] Reproductive period and risk of dementia in postmenopausal women.
- best-positive rank: ft_hardneg_dense=1, baseline=12  |  nDCG@10: ft_hardneg_dense=1.000, baseline=0.000
- ft_hardneg_dense #1 (cos 0.615): [6309659] Reproductive period and risk of dementia in postmenopausal women.
- ft_hardneg_dense #2 (cos 0.578): [37205759] Obesity occurring in apolipoprotein E-knockout mice has mild effects on fertility.
- ft_hardneg_dense #3 (cos 0.568): [51386222] Effects of Age, Sex, and Ethnicity on the Association Between Apolipoprotein E Genotype and Alzheimer Disease:

### Query 1180
**Claim:** The PRR MDA5 is a sensor of RNA virus infection.

**Judged positive(s):** [31272411] Immune signaling by RIG-I-like receptors.
- best-positive rank: ft_hardneg_dense=1, baseline=23  |  nDCG@10: ft_hardneg_dense=1.000, baseline=0.000
- ft_hardneg_dense #1 (cos 0.556): [31272411] Immune signaling by RIG-I-like receptors.
- ft_hardneg_dense #2 (cos 0.550): [44366096] RIG-I-mediated antiviral responses to single-stranded RNA bearing 5'-phosphates.
- ft_hardneg_dense #3 (cos 0.511): [2566674] Ribose 2′-O-methylation provides a molecular signature for the distinction of self and non-self mRNA dependent

### Query 1179
**Claim:** The PRR MDA5 has a central DExD/H RNA helices domain.

**Judged positive(s):** [31272411] Immune signaling by RIG-I-like receptors.
- best-positive rank: ft_hardneg_dense=1, baseline=None  |  nDCG@10: ft_hardneg_dense=1.000, baseline=0.000
- ft_hardneg_dense #1 (cos 0.515): [31272411] Immune signaling by RIG-I-like receptors.
- ft_hardneg_dense #2 (cos 0.487): [20725212] Chaperoning 5S RNA assembly.
- ft_hardneg_dense #3 (cos 0.476): [15419873] Cytosolic viral sensor RIG-I is a 5'-triphosphate-dependent translocase on double-stranded RNA.

### Query 1175
**Claim:** The PPR MDA5 has two N-terminal CARD domains.

**Judged positive(s):** [31272411] Immune signaling by RIG-I-like receptors.
- best-positive rank: ft_hardneg_dense=1, baseline=None  |  nDCG@10: ft_hardneg_dense=1.000, baseline=0.000
- ft_hardneg_dense #1 (cos 0.368): [31272411] Immune signaling by RIG-I-like receptors.
- ft_hardneg_dense #2 (cos 0.342): [20725212] Chaperoning 5S RNA assembly.
- ft_hardneg_dense #3 (cos 0.325): [3610282] Phase-plate cryo-EM structure of a biased agonist-bound human GLP-1 receptor–Gs complex

### Query 623
**Claim:** Individuals with low serum vitamin D concentrations have increased risk of multiple sclerosis.

**Judged positive(s):** [17000834] Past exposure to sun, skin phenotype, and risk of multiple sclerosis: case-control study.
- best-positive rank: ft_hardneg_dense=2, baseline=16  |  nDCG@10: ft_hardneg_dense=0.631, baseline=0.000
- ft_hardneg_dense #1 (cos 0.596): [22843838] [Vitamin D and latitude as environmental factors in multiple sclerosis].
- ft_hardneg_dense #2 (cos 0.592): [17000834] Past exposure to sun, skin phenotype, and risk of multiple sclerosis: case-control study.
- ft_hardneg_dense #3 (cos 0.539): [36960449] Estimation of the dietary requirement for vitamin D in healthy adults.
