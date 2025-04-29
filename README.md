modified_AT
This repository is a personal learning project aimed at analyzing the structure of the [(https://github.com/thuml/Anomaly-Transformer)] and experimenting with modifications.

Purpose
Understand the architecture of the original model
Test with changes
Record the results and link to blog explanations
Reference
For full report in Korean with figures, please refer to the blog post: [(https://mynews412.tistory.com/76)]

## Summary

1) The original Anomaly Transformer shows performance collapse when removing point adjust.  
2) Implemented modifications:  
   - Made prior a learnable parameter  
   - Applied unidirectional KL-divergence  
   - Added dimension restoration process  
   - And more
3) Modified model shows baseline-level performance on MSL/PSM datasets but underperforms on SMAP, suggesting need for data engineering improvements.  
4) Learnable prior_loss and alternating updates showed limited impact.  
5) ass-dis metric showed no significant improvement over S+P in fixed threshold settings. Dynamic thresholding may unlock potential.

---

## 1. Model Overview

**Multivariate Time-Series Anomaly Detection Model** (SOTA 2022)  
Core idea: Anomalies exhibit weakened global associations and strengthened local associations.

### Key Components:
- **Prior(P)**: Distance-based Gaussian kernel values 
- **Series(S)**: Attention scores based on reconstruction loss  
- **Loss Functions**:  
  `prior_loss(PL) = KL(P||S.detach()) + KL(S.detach()||P)`  
  `series_loss(SL) = KL(P.detach()||S) + KL(S||P.detach())`  
  `loss1 = rec_loss - k * SL` (Maximize SL)  
  `loss2 = rec_loss + k * PL` (Minimize PL)  

### Training Strategy:
- Minimax approach with simultaneous parameter updates  
- Test uses unidirectional KL-div:  
  `anomaly_score = softmax(-KL(S||P)*temp - KL(P||S)*temp) * rec_loss`

### Evaluation:
- Point adjust technique applied  
- [Original Paper](https://openreview.net/pdf?id=LzQQ89U1qm_)

---

## 2. Identified Issues

### 2.1 Association Discrepancy (AssDis)
- Theoretical justification for ass-dis (bidirectional KL-div) in anomaly detection remains unclear  
- No guarantee of divergence pattern changes in anomalies

### 2.2 Data Leakage
1. Threshold generation using test data  
2. Test data usage in validation

### 2.3 Minimax Implementation Flaws
- Parameter group conflicts due to simultaneous updates  
- Gradient interference from shared parameters

### 2.4 Bidirectional KL-divergence
- Causes identical PL/SL values → training failure

### 2.5 Window Processing
- Improper handling of sliding window outputs

### 2.6 Metric Selection
- Over-reliance on point-adjust metric

---

## 3. Experiments

### 3.1 Point Adjust Impact
| Dataset | Metric | With PA | Without PA | Δ |
|---------|--------|---------|------------|---|
| MSL | F-score | 0.9426 | 0.0201 | **-97.87%** |
| SMAP | F-score | 0.9644 | 0.0181 | **-98.12%** |
| PSM | F-score | 0.9764 | 0.0223 | **-97.72%** |

### 3.2 Model Modifications
1. Alternative metrics (S+P vs ass-dis)  
2. Parameter group separation + alternating updates  
3. Data leakage prevention  
4. Unidirectional KL-div implementation  
5. Window dimension restoration  
6. Learning rate scheduling  
7. Layer reduction (3→2)

### 3.3 Benchmark Comparison (LSTM-AE)
Architecture
StackedLSTMAE(
encoder: LSTM(55→64, 3 layers),
decoder: LSTM(64→64, 3 layers),
fc: Linear(64→55)
)


### 3.4 Performance Metrics
| Dataset | Model | AUC-ROC | AUC-PR |
|---------|-------|---------|--------|
| MSL | AT-ass-dis | 0.7212 | 0.2526 |
| MSL | AT-S+P | 0.7236 | 0.2571 |
| PSM | LSTM-AE | 0.7779 | 0.5560 |

### 3.5 Metric Comparison
- **MSL**: Significant score divergence at 38000-40000 indices  
- **SMAP/PSM**: Minimal practical difference despite statistical significance

### 3.6 Epoch Analysis
![](https://blog.kakaocdn.net/dn/OUhab/btsNB58e9tA/kUEEYt48v8J2ktxiol5MNK/img.png)  
(Epoch progression shows metric convergence)

---

## 4. Conclusion

1. Successfully decoupled model from point adjust dependency  
2. Achieved baseline performance on MSL/PSM through architectural improvements  
3. SMAP dataset requires fundamental data engineering changes  
4. Learnable prior parameters showed limited impact  
5. Dynamic thresholding may unlock ass-dis potential  


---

