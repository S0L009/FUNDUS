<div align="center">

# AutoFundus

![Fundus Image](flow.png)

</div>



## Pipeline:

1. **Region of Interest Extraction:** Utilizing computer vision methods to isolate the ROI from the entire Fundus Image.
2. **Optic Disc Segmentation:** Isolating the optic disc from the cropped fundus images.
3. **Glaucoma Classification:** Differentiating between glaucoma suspects and healthy individuals
   
## Results:

| Prediction | Groundtruth |
|------------|-------------|
| ![Prediction Image](ppred_1.png) | ![Groundtruth Image](llabel_1.png) |

## Metrics
- Accuracy: 94.79
- Precision: 97.07
- Recall: 92.50
- F1 Score: 94.73
- Dice Score - 94.40
- IoU: 90.24
